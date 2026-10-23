"""Chat-completions transport with explicit credentials and no automatic retry."""
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request


class ReviewError(ValueError):
    """A review request or response cannot be used for scoring."""


def require(condition, message):
    if not condition:
        raise ReviewError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result

    def constant(_):
        raise ReviewError('Non-finite JSON number')

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError) as exc:
        raise ReviewError('Expected a complete, strict JSON document') from exc


def load_config(path):
    config = strict_json(path.read_text(encoding='utf-8-sig'))
    require(isinstance(config, dict), 'Configuration must be an object')
    fields = {'api_key', 'base_url', 'model', 'timeout_seconds', 'max_tokens',
              'temperature', 'token_limit_field', 'json_mode'}
    require(set(config) <= fields, 'Unknown configuration field')
    for key in ('api_key', 'base_url', 'model'):
        override = os.environ.get('DTB_' + key.upper())
        if override:
            config[key] = override
        require(isinstance(config.get(key), str) and config[key].strip(),
                f'Fill {key} in the configuration or DTB_{key.upper()}')
        config[key] = config[key].strip()
    require(not re.search(r'[\x00-\x1f\x7f]', config['api_key']), 'Invalid API key characters')
    config.setdefault('timeout_seconds', 180)
    config.setdefault('max_tokens', 8192)
    config.setdefault('temperature', 0)
    config.setdefault('token_limit_field', 'max_tokens')
    config.setdefault('json_mode', False)
    require(type(config['max_tokens']) is int and config['max_tokens'] > 0, 'Invalid max_tokens')
    require(type(config['timeout_seconds']) in (int, float)
            and math.isfinite(config['timeout_seconds']) and config['timeout_seconds'] > 0,
            'Invalid timeout_seconds')
    require(type(config['temperature']) in (int, float)
            and math.isfinite(config['temperature']) and 0 <= config['temperature'] <= 2,
            'Invalid temperature')
    require(config['token_limit_field'] in ('max_tokens', 'max_completion_tokens'),
            'Unsupported token_limit_field')
    require(type(config['json_mode']) is bool, 'json_mode must be Boolean')
    return config


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatClient:
    def __init__(self, config):
        self.config = config
        require(not re.search(r'\s', config['base_url']), 'Base URL must not contain whitespace')
        url = urllib.parse.urlsplit(config['base_url'])
        require(url.scheme == 'https' and bool(url.hostname), 'Base URL must use HTTPS')
        require(not url.username and not url.password and not url.query and not url.fragment,
                'Base URL must not contain credentials, query parameters or a fragment')
        base = config['base_url'].rstrip('/')
        self.endpoint = base if base.endswith('/chat/completions') else base + '/chat/completions'
        self.opener = urllib.request.build_opener(NoRedirect())

    @property
    def settings(self):
        return {key: self.config[key] for key in
                ('model', 'max_tokens', 'temperature', 'token_limit_field', 'json_mode')}

    def payload(self, messages):
        result = {'model': self.config['model'], 'messages': messages, 'stream': False,
                  'temperature': self.config['temperature'],
                  self.config['token_limit_field']: self.config['max_tokens']}
        if self.config['json_mode']:
            result['response_format'] = {'type': 'json_object'}
        return result

    def redact(self, value):
        """Remove configured credentials/endpoints from any saved server response."""
        if isinstance(value, dict):
            return {self.redact(k): ('[REDACTED]' if k.lower() in
                    {'api_key', 'base_url', 'authorization', 'access_token'} else self.redact(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            for secret in (self.config['api_key'], self.endpoint, self.config['base_url']):
                if secret:
                    value = value.replace(secret, '[REDACTED]')
        return value

    def complete(self, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        request = urllib.request.Request(self.endpoint, data=body, method='POST', headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.config['api_key']})
        try:
            with self.opener.open(request, timeout=self.config['timeout_seconds']) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
                require(len(raw) <= 32 * 1024 * 1024, 'Response exceeds the 32 MiB safety limit')
                text = raw.decode('utf-8')
                try:
                    envelope = strict_json(text)
                except ReviewError:
                    envelope = {'transport_raw_text': text, 'transport_parse_error': True}
                return self.redact(envelope)
        except urllib.error.HTTPError as exc:
            raise ReviewError(f'HTTP {exc.code}; no automatic retry; usage may be unknown') from None
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeError):
            raise ReviewError('Transport failure; no automatic retry; usage may be unknown') from None


def response_document(response):
    require(isinstance(response, dict), 'Response envelope must be an object')
    choices = response.get('choices')
    require(isinstance(choices, list) and len(choices) == 1, 'Expected exactly one response choice')
    choice = choices[0]
    require(isinstance(choice, dict) and choice.get('finish_reason') == 'stop',
            'Incomplete or non-stop response; do not import partial decisions')
    message = choice.get('message')
    require(isinstance(message, dict) and not message.get('refusal'), 'Missing or refused answer')
    require(not message.get('tool_calls'), 'Tool calls are not review decisions')
    content = message.get('content')
    require(isinstance(content, str) and content.strip(), 'Empty or unsupported answer content')
    document = strict_json(content)
    require(isinstance(document, dict), 'Review answer must be a JSON object')
    return document
