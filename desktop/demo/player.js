/* Cancellable, bounded playback; no network, model client or research executor. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.DemoPlayer = api;
})(globalThis, function () {
  'use strict';
  const MAX_DELAY = 5000;
  function wait(ms, signal) {
    if (!Number.isFinite(ms) || ms < 0 || ms > MAX_DELAY) return Promise.reject(new Error('Invalid demo wait'));
    return new Promise((resolve,reject) => {
      const abort=()=>{clearTimeout(timer); signal?.removeEventListener('abort',abort); reject(new DOMException('Stopped','AbortError'));};
      const timer=setTimeout(()=>{signal?.removeEventListener('abort',abort);resolve();},ms);
      if(signal?.aborted) abort(); else signal?.addEventListener('abort',abort,{once:true});
    });
  }
  async function play(events,{signal,onEvent,onProgress=()=>{},sleep=wait}) {
    for(let i=0;i<events.length;i++) {
      if(signal?.aborted) throw new DOMException('Stopped','AbortError');
      const event=events[i];
      await onEvent(event,ms=>sleep(ms,signal));
      if(signal?.aborted) throw new DOMException('Stopped','AbortError');
      onProgress(i+1,events.length);
    }
  }
  return {MAX_DELAY,wait,play};
});
