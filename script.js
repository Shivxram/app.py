
function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
async function typeInto(id, text) {
  const el = document.getElementById(id);
  if(!el) return;
  el.innerHTML = '';
  for(let i=0;i<text.length;i++){
    el.innerHTML += text[i];
    await sleep(6);
  }
}
