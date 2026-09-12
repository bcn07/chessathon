// Paste into the browser console on https://aichessathon.com/dashboard while logged in.
// Fetches every agent log linked on the page using your session and downloads ONE file.
// v2: 2 s between requests and a retry on HTTP 429 — the first run was rate-limited after ~59.
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const urls = [...new Set([...document.querySelectorAll('a[href*="/api/platform/logs/"]')].map(a => a.href))]
    .filter(u => !u.endsWith('/export'));
  console.log(`found ${urls.length} log links`);
  const parts = [];
  for (let i = 0; i < urls.length; i++) {
    let r, t = '';
    for (let attempt = 1; attempt <= 4; attempt++) {
      r = await fetch(urls[i], { credentials: 'include' });
      if (r.status !== 429) { t = await r.text(); break; }
      console.warn(`${i + 1}: 429, waiting ${10 * attempt}s (attempt ${attempt})`);
      await sleep(10000 * attempt);
    }
    parts.push(`===== LOG ${i + 1}/${urls.length} ${urls[i]} (HTTP ${r.status}) =====\n${t}\n`);
    console.log(`${i + 1}/${urls.length} HTTP ${r.status} (${t.length} bytes)`);
    await sleep(2000);
  }
  const blob = new Blob([parts.join('\n')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'aichessathon-all-logs.txt';
  document.body.appendChild(a); a.click(); a.remove();
  console.log('done — check your Downloads for aichessathon-all-logs.txt');
})();
