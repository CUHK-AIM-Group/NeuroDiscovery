"""Bounded exception metadata; never persist provider messages or request payloads."""

from datetime import datetime, timezone
import errno
import re
import socket
import ssl


def exception_diagnostic(exc, operation='model_request'):
    chain = []
    seen = set()
    current = exc
    while isinstance(current, BaseException) and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        name = type(current).__name__
        entry = {'type': name if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,79}', name) else 'Exception'}
        status = getattr(current, 'status_code', None)
        if type(status) is int and 100 <= status <= 599:
            entry['http_status'] = status
        code = getattr(current, 'errno', None)
        if isinstance(current, OSError) and type(code) is int:
            entry['errno'] = code
            if not isinstance(current, (ssl.SSLError, socket.gaierror)) and code in errno.errorcode:
                entry['errno_name'] = errno.errorcode[code]
        if isinstance(current, ssl.SSLError):
            reason = getattr(current, 'reason', None)
            reasons = {'UNEXPECTED_EOF_WHILE_READING', 'CERTIFICATE_VERIFY_FAILED', 'WRONG_VERSION_NUMBER',
                       'TLSV1_ALERT_INTERNAL_ERROR', 'TLSV1_ALERT_PROTOCOL_VERSION', 'SSLV3_ALERT_HANDSHAKE_FAILURE'}
            if reason in reasons:
                entry['tls_reason'] = reason
            elif isinstance(current, ssl.SSLEOFError):
                entry['tls_reason'] = 'UNEXPECTED_EOF_WHILE_READING'
            entry['category'] = 'tls'
        elif isinstance(current, socket.gaierror):
            entry['category'] = 'dns'
        elif name in {'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout', 'APITimeoutError', 'TimeoutError'}:
            entry['category'] = 'timeout'
        elif name in {'ConnectError', 'APIConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'RemoteProtocolError'}:
            entry['category'] = 'connection'
        elif 'http_status' in entry:
            entry['category'] = 'http'
        chain.append(entry)
        following = current.__cause__
        relation = 'cause'
        if following is None and not current.__suppress_context__:
            following = current.__context__
            relation = 'context'
        if isinstance(following, BaseException) and id(following) not in seen and len(chain) < 6:
            entry['next_relation'] = relation
        current = following
    return {'timestamp': datetime.now(timezone.utc).isoformat(), 'operation': operation,
            'exception_chain': chain, 'category': next((item['category'] for item in reversed(chain) if 'category' in item), 'unknown'),
            'http_status': next((item['http_status'] for item in chain if 'http_status' in item), None),
            'details_policy': 'Structured metadata only; exception messages, URLs, headers, bodies and credentials omitted.'}
