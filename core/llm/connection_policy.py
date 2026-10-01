"""Retry only identifiable pre-request connection failures, never ambiguous delivery."""

import ssl

import httpx


def retryable_connection_failure(error):
    chain = []
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen and len(chain) < 12:
        seen.add(id(error))
        chain.append(error)
        if getattr(error, 'response', None) is not None or getattr(error, 'status_code', None) is not None:
            return False
        if type(error).__name__ == 'IncompleteModelResponse' or isinstance(error, (
                httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.WriteError,
                httpx.RemoteProtocolError, ssl.SSLCertVerificationError)):
            return False
        error = error.__cause__
    return any(isinstance(item, (httpx.ConnectTimeout, httpx.ConnectError)) for item in chain)
