'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const API = (window.PLATEAU_API || '').replace(/\/$/, '');
let modelLayers = {};
let backendLabel = '';
let localLibrary=false;
const historySelection=new Set();
const definedMetric=(row,metric)=>PlateauRecords.valid(row?.values?.[metric]);
const presets = [
  ['The house was big','The house was in'],
  ['The capital of France','The capital of Germany'],
  ['She was very happy','She was very sad'],
  ['The weather was hot','The weather was cold'],
  ['He opened the door','He opened the book'],
  ['I would like some coffee','After a long walk, she asked for tea'],
  ['The answer is yes','The answer is no'],
  ['The movie was good','The movie was bad'],
  ['The capital of France is Paris. The capital of Japan is','The capital of France is Paris. The capital of Germany is'],
];
let result = null, busy = false, currentJob = null, presetIndex = 3;
let toastTimer, tokenizeTimer, tokenizeRequest = 0;
function formatDate(value, full=false) {
  const iso = new Date(value).toISOString();
  return full ? iso.slice(0, 19).replace('T', ' ') + ' UTC' : iso.slice(0, 10);
}
async function api(path, body, headers={}) {
  const response = await fetch(API+path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed. Please try again.');
  return data;
}
function toast(text) { $('toast').textContent=text; $('toast').classList.remove('hidden'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),3500); }
function status(text, progress=0, error=false) { $('status').classList.remove('hidden'); $('status').classList.toggle('error',error); $('status').classList.toggle('indeterminate',progress===null); $('status-text').textContent=text; $('progress').style.width=progress===null?'30%':`${Math.max(0,Math.min(1,progress))*100}%`; }
async function loadConfig() {
  const config=await api('/api/config');
  modelLayers=Object.fromEntries(config.models.map(m=>[m.id,m.layers]));
  $('model').innerHTML=[...new Set(config.models.map(m=>m.family))].map(family=>
    `<optgroup label="${esc(family)}">`+config.models.filter(m=>m.family===family).map(m=>
      `<option value="${esc(m.id)}" title="${esc(m.id)}">${esc(m.label)}${m.running===false?' · not running':''}</option>`).join('')+'</optgroup>').join('');
  $('model').value=config.default_model;
  requiresKey=config.requires_key;
  localLibrary=!!config.local_library;
  $('classic-link').classList.toggle('hidden',!localLibrary);
  $('history-caption').textContent=localLibrary?'Local runs are saved on disk. Browser imports are also shown. Saved examples and notes remain in Local collections.':'Runs are stored in this browser. Export JSON, JSONL or CSV to keep or share them.';
  $('key-field').classList.toggle('hidden',!config.remote);
  sharedAccess=config.shared_access;
  $('code-note').classList.toggle('hidden',!sharedAccess);
  $('key-label').textContent=sharedAccess?'NDIF API key or lab access code':'NDIF API key';
  $('ndif-key').placeholder=sharedAccess?'Your key (xxxxxxxx-xxxx-…) or the lab code':'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';
  keyState();
  backendLabel=config.remote?'NDIF remote inference':'Local inference';
  $('backend-badge').innerHTML=`<span class="dot"></span> ${config.remote?'NDIF':'LOCAL'}`;
  $('backend-name').textContent=backendLabel;
  $('backend-note').textContent=(config.remote?'Runs on the National Deep Inference Fabric.':'Runs on the server\'s own hardware.')+(localLibrary?' Runs are saved on this computer.':' Recent results are saved in this browser only.');
  return config;
}
function settings() { return {model:$('model').value, sequence_a:$('sequence-a').value, sequence_b:$('sequence-b').value, interpolation:$('interpolation').value, patch_layer:Number($('patch-layer').value), patch_position:$('patch-position').value, steps:Number($('steps').value), context:$('context').value}; }
function suffixMode() { return $('patch-position').value==='different_suffix'; }
function patchLabel(settings) { return settings.patch_position==='different_suffix'?'First difference → end':'Final token only'; }
function patchNote() {
  const suffix=suffixMode();
  $('context-field').classList.toggle('hidden',suffix);
  $('context').disabled=busy || suffix;
  $('patch-note').textContent=suffix
    ? 'Equal token counts required. Pair positions directly and interpolate every token from the first difference onward, including later matching tokens. The same t is used for each pair; the shared prefix stays fixed.'
    : 'Interpolate only the final token from each input. Prefixes and token counts may differ; choose A or B as the fixed context.';
}
function remember() { try{localStorage.setItem('plateau-draft-v1',JSON.stringify(settings()));}catch{} }
const DEFAULT_PATCH_LAYER=0;  // clamped to the model's last layer
function layers(selected=DEFAULT_PATCH_LAYER) {
  const count=modelLayers[$('model').value];
  $('patch-layer').max=String(count-1);
  $('patch-layer').value=String(Math.max(0,Math.min(selected,count-1)));
  $('layer-max').textContent=String(count-1);
  layerNote();
}
function layerNote() {
  const layer=Number($('patch-layer').value), count=modelLayers[$('model').value];
  $('layer-value').textContent=`After layer ${layer}`;
  $('patch-layer').setAttribute('aria-valuetext',`After layer ${layer}`);
  $('layer-note').textContent=`Hidden space: interpolate the selected token states after layer ${layer} (resid_post), then ${layer===count-1?'apply final normalization and the output head':`continue from layer ${layer+1}`}. Layers are numbered 0–${count-1}.`;
  patchNote();
}
function setForm(record) {
  const s=record.settings || record;
  $('sequence-a').value=record.sequence_a;
  $('sequence-b').value=record.sequence_b;
  if(modelLayers[record.model]) $('model').value=record.model;  // a stored run's model may no longer be offered
  $('patch-position').value=s.patch_position || 'last_token';
  layers(s.patch_layer ?? DEFAULT_PATCH_LAYER);
  $('interpolation').value=s.interpolation || 'slerp';
  $('context').value=s.context || 'a';
  $('steps').value=String(s.steps || 41);
  if(!$('steps').value) $('steps').value='41';
}
function setBusy(value) {
  busy=value;
  ['model','interpolation','patch-layer','patch-position','steps','context','sequence-a','sequence-b','swap','next-preset','run'].forEach(id=>$(id).disabled=value);
  patchNote();
  document.querySelectorAll('[data-preset]').forEach(el=>el.disabled=value);
  $('cancel').classList.toggle('hidden',!value);
  $('run').innerHTML=value?'<span>◌</span> Computing…':'<span>▶</span> Run experiment <kbd>⌘/Ctrl ↵</kbd>';
}
function invalidate() {
  result=null;
  $('metric-overview').innerHTML=''; $('export-result').disabled=true;
  ['result-meta','token-details','effect-section','inspect'].forEach(id=>$(id).classList.add('hidden'));
  scheduleTokenize();
  layerNote();
  $('status').classList.add('hidden');
  $('effect-placeholder').classList.remove('hidden');
  for(const side of ['a','b']) { $('prediction-'+side).textContent='Run an experiment to see the continuation'; $('prediction-'+side).classList.add('muted'); }
  remember();
}
function preset(index) { if(busy)return; $('sequence-a').value=presets[index][0];$('sequence-b').value=presets[index][1];invalidate(); }
function tokenText(value) { return String(value).replace(/ /g,'␣').replace(/\n/g,'↵').replace(/\t/g,'⇥').replace(/\r/g,'␍'); }
// Live preview: tokenize both inputs with the selected model's tokenizer while typing.
function scheduleTokenize() { clearTimeout(tokenizeTimer); tokenizeTimer=setTimeout(tokenizeInputs,200); }
async function tokenizeInputs() {
  const id=++tokenizeRequest, s=settings();
  try {
    const preview=await api('/api/tokenize',{model:s.model,sequence_a:s.sequence_a,sequence_b:s.sequence_b,patch_position:s.patch_position});
    if(id===tokenizeRequest && !result) renderInputTokens(preview.tokens,preview.patch_starts,s.patch_position==='different_suffix',preview.max_tokens);
  } catch(error) {
    if(id===tokenizeRequest) for(const side of ['a','b']) $('token-summary-'+side).textContent='Tokenization unavailable: '+error.message;
  }
}
function renderInputTokens(inputTokens, starts, suffix, maxTokens=256) {
  $('tokenization-note').textContent=suffix
    ? 'Highlighted: every token from the first difference through the end, including matching tokens after it. Each hidden-state pair is interpolated with the same t. Hover for token IDs.'
    : 'Highlighted: the final token whose state is interpolated. Hover over a token to see its ID.';
  const unequal=suffix && inputTokens[0].length!==inputTokens[1].length && inputTokens[0].length && inputTokens[1].length;
  inputTokens.forEach((tokens,i)=>{
    const side=i?'b':'a';
    const start=starts ? starts[i] : tokens.length;
    $('count-'+side).textContent=`${tokens.length} token${tokens.length===1?'':'s'}`;
    $('count-'+side).classList.toggle('over-limit',tokens.length>maxTokens);
    $('token-summary-'+side).textContent=!tokens.length ? 'No tokens yet'
      : tokens.length>maxTokens ? `${tokens.length} tokens · the limit is ${maxTokens}`
      : unequal ? `${tokens.length} tokens · suffix mode needs equal counts (A: ${inputTokens[0].length}, B: ${inputTokens[1].length})`
      : !starts ? `${tokens.length} tokens`
      : `${tokens.length} tokens · interpolating ${tokens.length-start} at ${start===tokens.length-1?`position ${start}`:`positions ${start}–${tokens.length-1}`}`;
    $('input-tokens-'+side).innerHTML=tokens.map((token,index)=>{
      const patched=index>=start, first=suffix && index===start;
      const description=`Position ${index} · token ID ${token.id}${patched?' · interpolated position':''}${first?' · first difference':''}`;
      return `<span class="input-token${patched?' final-token':''}${first?' first-difference':''}" role="listitem" title="${esc(description)}"><small>${index}</small><code>${esc(tokenText(token.text))}</code>${first?'<span class="token-target">First difference</span>':patched && index===tokens.length-1?'<span class="token-target">Final</span>':''}<span class="sr-only">${esc(description)}</span></span>`;
    }).join('');
  });
}
function formatL2(value) {
  if(!Number.isFinite(value))return '—';
  if(value===0)return '0';
  return value<0.001 || value>=100000 ? value.toExponential(3) : value.toLocaleString('en-US',{maximumFractionDigits:4});
}
// Effect figure: one metric vs t for the selected layers (left), per-layer list (right).
const RAMP=['#a594e8','#7f6ad6','#5a44b8','#3b2789'], LOGITS_COLOR='#427eaa';
let effectSelection=new Set(), effectPreset='representative';
// Categorical slots in fixed order (reference palette); tokens past the eighth fold into "Other".
const TOKEN_COLORS=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'], OTHER_COLOR='#9a9ca6';
// Runs of equal argmax token along t; boundaries sit halfway between the samples where it changes.
function tokenSegments(record) {
  const t=record.effect.t, preds=record.path_predictions, order=new Map(), segments=[];
  preds.forEach((p,i)=>{
    if(!order.has(p.token_id)) order.set(p.token_id,order.size);
    const start=i===0?t[0]:(t[i-1]+t[i])/2;
    if(segments.length && segments[segments.length-1].id===p.token_id) return;
    if(segments.length) segments[segments.length-1].end=start;
    segments.push({id:p.token_id,token:p.token,start,end:t[t.length-1]});
  });
  const color=id=>order.get(id)<TOKEN_COLORS.length?TOKEN_COLORS[order.get(id)]:OTHER_COLOR;
  return {segments:segments.map(s=>({...s,color:color(s.id)})),
    tokens:[...order.keys()].map(id=>({id,token:preds.find(p=>p.token_id===id).token,color:color(id),other:order.get(id)>=TOKEN_COLORS.length}))};
}
function hexMix(a,b,f) { const p=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)); const [x,y]=[p(a),p(b)]; return '#'+x.map((v,i)=>Math.round(v+(y[i]-v)*f).toString(16).padStart(2,'0')).join(''); }
// Color follows the layer's depth (never its rank in the selection): light = shallow, dark = deep.
function effectColor(row, nLayers) {
  if(row.key==='logits') return LOGITS_COLOR;
  const f=nLayers>1?row.layer/(nLayers-1):1, x=f*(RAMP.length-1), i=Math.min(RAMP.length-2,Math.floor(x));
  return hexMix(RAMP[i],RAMP[i+1],x-i);
}
function effectPresetKeys(record, preset) {
  const rows=record.effect.rows.filter(row=>row.defined), layers=rows.filter(row=>row.key!=='logits');
  if(preset==='last') return layers.slice(-1).map(row=>row.key);
  if(preset==='logits') return ['logits'];
  if(preset==='all') return rows.map(row=>row.key);
  if(preset==='none') return [];
  if(preset==='even10') return [...new Set(Array.from({length:Math.min(10,layers.length)},(_,i)=>layers[Math.round(i*(layers.length-1)/Math.max(1,Math.min(10,layers.length)-1))].key))];
  return [...(record.settings.representative_layers || []).map(String),'logits'].filter(key=>rows.some(row=>row.key===key));
}
function applyEffectPreset(preset) {
  effectPreset=preset; $('effect-preset').value=preset;
  effectSelection=new Set(effectPresetKeys(result,preset));
  renderEffect();
}
// Across-layers figure: endpoint L2 or a metric's plateau score per recorded block.
function layerQuantities(record) {
  return [{id:'total_length',label:'Total sampled path L2',value:row=>row.total_length,reference:null},{id:'endpoint_l2',label:'Endpoint L2',value:row=>row.endpoint_l2,reference:null},
    ...record.effect.metrics.filter(m=>m.plateau_score).map(m=>({id:'plateau:'+m.id,label:`Plateau score Δt · ${m.label}`,
      value:row=>row.plateau_score[m.id],reference:{value:0.8,label:m.id==='c'?'c = t':'d = t'}}))];
}
function renderLayerPlot() {
  if(!result)return;
  const quantity=layerQuantities(result).find(q=>q.id===$('layer-quantity').value) || layerQuantities(result)[0];
  const rows=result.effect.rows.filter(row=>row.key!=='logits');
  const points=rows.map(row=>({row,v:quantity.value(row)}));
  const values=points.map(p=>p.v).filter(v=>v!=null && Number.isFinite(v));
  const w=960,h=300,left=62,right=w-18,top=24,bottom=h-44;
  if(!rows.length){$('layer-plot').textContent='No layer readouts were saved in this result.';return;}
  const first=rows[0].layer,last=rows[rows.length-1].layer;
  let lo=Math.min(0,...values),hi=Math.max(...values,quantity.reference?quantity.reference.value:-Infinity)*1.08;
  if(!Number.isFinite(hi) || hi<=lo) hi=lo+1;
  const x=layer=>left+(last===first?.5:(layer-first)/(last-first))*(right-left), y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  const step=Math.max(1,Math.ceil((last-first+1)/16)), ticks=rows.filter(r=>(r.layer-first)%step===0 || r.layer===last).map(r=>r.layer);
  // Line segments break at undefined layers.
  const segments=[];let current=[];
  points.forEach(p=>{if(p.v==null || !Number.isFinite(p.v)){if(current.length)segments.push(current);current=[];}else current.push(p);});
  if(current.length)segments.push(current);
  const patch=result.settings.patch_layer, label=`${quantity.label} for layers ${first} to ${last}.`;
  $('layer-plot').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>
    ${[0,.25,.5,.75,1].map(f=>{const v=lo+f*(hi-lo);return `<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#eeeff2"/><text x="${left-9}" y="${y(v)+3.5}" text-anchor="end" fill="#8a8d99" font-size="11">${formatL2(+v.toPrecision(3))}</text>`;}).join('')}
    ${ticks.map(layer=>`<text x="${x(layer)}" y="${bottom+19}" text-anchor="middle" fill="#8a8d99" font-size="11">${layer}</text>`).join('')}
    <text x="${(left+right)/2}" y="${h-8}" text-anchor="middle" fill="#7d808d" font-size="11">Layer (block output, 0-based)</text>
    <text x="${left}" y="13" fill="#7d808d" font-size="11">${esc(quantity.label)}</text>
    <line x1="${x(patch)}" x2="${x(patch)}" y1="${top}" y2="${bottom}" stroke="#b7aacd" stroke-dasharray="3 4"/><text x="${x(patch)+5}" y="${top+9}" fill="#7762c7" font-size="10">Patch · layer ${patch}</text>
    ${quantity.reference?`<line x1="${left}" x2="${right}" y1="${y(quantity.reference.value)}" y2="${y(quantity.reference.value)}" stroke="#c2c5cf" stroke-dasharray="4 5"/><text x="${right}" y="${y(quantity.reference.value)-5}" text-anchor="end" fill="#9a9ca6" font-size="10">${quantity.reference.label} (${quantity.reference.value})</text>`:''}
    ${segments.map(seg=>`<polyline points="${seg.map(p=>`${x(p.row.layer).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ')}" fill="none" stroke="#7762c7" stroke-width="2" stroke-linejoin="round"/>`).join('')}
    ${points.filter(p=>p.v!=null && Number.isFinite(p.v)).map(p=>`<circle cx="${x(p.row.layer)}" cy="${y(p.v)}" r="4" fill="#7762c7" stroke="#fff" stroke-width="2"/>`).join('')}
    <line id="layer-crosshair" y1="${top}" y2="${bottom}" stroke="#9a9ca6" visibility="hidden"/>
    <rect id="layer-hit" x="${left-10}" y="${top}" width="${right-left+20}" height="${bottom-top}" fill="transparent"/>
  </svg><div id="layer-tooltip" class="effect-tooltip hidden" role="status"></div>`;
  const svg=$('layer-plot').querySelector('svg'), tip=$('layer-tooltip'), cross=$('layer-crosshair');
  $('layer-hit').addEventListener('pointermove',event=>{
    const box=svg.getBoundingClientRect(), vx=(event.clientX-box.left)/box.width*w;
    const p=points.reduce((best,q)=>Math.abs(x(q.row.layer)-vx)<Math.abs(x(best.row.layer)-vx)?q:best,points[0]);
    cross.setAttribute('x1',x(p.row.layer));cross.setAttribute('x2',x(p.row.layer));cross.setAttribute('visibility','visible');
    const head=document.createElement('div'), line=document.createElement('div'), val=document.createElement('b'), name=document.createElement('span');
    head.className='effect-tooltip-head'; head.textContent=p.row.label+(p.row.patched?' · patched':'');
    val.textContent=p.v==null?'undefined':formatL2(p.v); name.textContent=quantity.label;
    line.append(val,name); tip.replaceChildren(head,line); tip.classList.remove('hidden');
    const px=x(p.row.layer)/w*box.width;
    tip.style.left=`${px>box.width/2?px-tip.offsetWidth-12:px+12}px`; tip.style.top='12px';
  });
  $('layer-hit').addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');tip.classList.add('hidden');});
}
function renderEffect() {
  if(!result)return;
  const effect=result.effect, metric=$('effect-metric').value, info=effect.metrics.find(m=>m.id===metric);
  const nLayers=result.l2_distances.layers.length, t=effect.t;
  const shown=effect.rows.filter(row=>definedMetric(row,metric) && effectSelection.has(row.key));
  // Plot
  const overlay=$('effect-overlay').value==='next_token' && result.path_predictions.length===effect.t.length?tokenSegments(result):null;
  const w=720,h=overlay?458:430,left=52,right=w-18,top=overlay?46:18,bottom=h-46;
  // The metric's range is always shown and extended to fit values outside it (e.g. overshoot).
  const values=shown.flatMap(r=>r.values[metric]), [r0,r1]=info.range || [Infinity,-Infinity];
  let lo=Math.min(r0,...values), hi=Math.max(r1,...values);
  if(!Number.isFinite(lo) || !Number.isFinite(hi)){lo=0;hi=1;} if(hi===lo){hi=lo+1;}
  const x=v=>left+v*(right-left), y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  const ticks=[0,.25,.5,.75,1];
  const label=`${info.label} against interpolation coefficient t for ${shown.length} selected output${shown.length===1?'':'s'}.`;
  $('effect-plot').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>
    ${ticks.map(f=>`<line x1="${left}" x2="${right}" y1="${y(lo+f*(hi-lo))}" y2="${y(lo+f*(hi-lo))}" stroke="#eeeff2"/><text x="${left-9}" y="${y(lo+f*(hi-lo))+3.5}" text-anchor="end" fill="#8a8d99" font-size="11">${+(lo+f*(hi-lo)).toFixed(2)}</text><text x="${x(f)}" y="${bottom+19}" text-anchor="middle" fill="#8a8d99" font-size="11">${f}</text>`).join('')}
    ${overlay?overlay.segments.map(s=>`<rect x="${x(s.start)}" y="${top}" width="${Math.max(0,x(s.end)-x(s.start))}" height="${bottom-top}" fill="${s.color}" fill-opacity=".07"/>`).join('')+
      overlay.segments.map(s=>{const sw=x(s.end)-x(s.start), text=tokenText(s.token), fits=text.length*6.6+10<sw;
        return `<g><title>${esc(`Next token ${JSON.stringify(s.token)} for t ≈ ${s.start.toFixed(3)}–${s.end.toFixed(3)}`)}</title><rect x="${x(s.start)+1}" y="${top-26}" width="${Math.max(0,sw-2)}" height="20" rx="3" fill="${s.color}" fill-opacity=".16"/><rect x="${x(s.start)+1}" y="${top-26}" width="${Math.max(0,sw-2)}" height="3" rx="1.5" fill="${s.color}"/>${fits?`<text x="${(x(s.start)+x(s.end))/2}" y="${top-11}" text-anchor="middle" fill="#3d3f4a" font-size="11" font-family="SFMono-Regular,Consolas,monospace">${esc(text)}</text>`:''}</g>`;}).join(''):''}
    <text x="${(left+right)/2}" y="${h-8}" text-anchor="middle" fill="#7d808d" font-size="11">Interpolation coefficient t</text>
    <text x="${left}" y="11" fill="#7d808d" font-size="11">${esc(info.label)}${overlay?' · strip: next-token prediction':''}</text>
    <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#c2c5cf" stroke-dasharray="4 5"/>
    ${lo<0?`<line x1="${left}" x2="${right}" y1="${y(0)}" y2="${y(0)}" stroke="#d6d8df"/>`:''}${hi>1?`<line x1="${left}" x2="${right}" y1="${y(1)}" y2="${y(1)}" stroke="#d6d8df"/>`:''}
    ${shown.map(row=>`<polyline points="${row.values[metric].map((v,i)=>`${x(t[i]).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}" fill="none" stroke="${effectColor(row,nLayers)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`).join('')}
    <line id="effect-crosshair" y1="${top}" y2="${bottom}" stroke="#9a9ca6" stroke-width="1" visibility="hidden"/>
    <rect id="effect-hit" x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="transparent"/>
  </svg><div id="effect-tooltip" class="effect-tooltip hidden" role="status"></div>`;
  $('effect-empty').classList.toggle('hidden',shown.length>0);
  $('effect-legend').innerHTML=shown.map(row=>`<span><i style="background:${effectColor(row,nLayers)}"></i>${esc(row.label)}</span>`).join('')+
    (overlay?`<span class="legend-break">Next token:</span>`+overlay.tokens.filter(tok=>!tok.other).map(tok=>`<span><i class="token-swatch" style="background:${tok.color}"></i><code>${esc(tokenText(tok.token))}</code></span>`).join('')+
      (overlay.tokens.some(tok=>tok.other)?`<span><i class="token-swatch" style="background:${OTHER_COLOR}"></i>Other (${overlay.tokens.filter(tok=>tok.other).length})</span>`:''):'');
  // Crosshair + tooltip: snap to the nearest sample, list every shown series there.
  const svg=$('effect-plot').querySelector('svg'), tip=$('effect-tooltip'), cross=$('effect-crosshair');
  const move=event=>{
    if(!shown.length)return;
    const box=svg.getBoundingClientRect(), vx=(event.clientX-box.left)/box.width*w;
    const i=t.reduce((best,v,j)=>Math.abs(x(v)-vx)<Math.abs(x(t[best])-vx)?j:best,0);
    cross.setAttribute('x1',x(t[i]));cross.setAttribute('x2',x(t[i]));cross.setAttribute('visibility','visible');
    tip.replaceChildren();
    const head=document.createElement('div');head.className='effect-tooltip-head';head.textContent=`t = ${t[i].toFixed(3)}`;tip.append(head);
    if(overlay){const next=document.createElement('div'), code=document.createElement('code');next.className='effect-tooltip-head';code.textContent=tokenText(result.path_predictions[i].token);next.append('Next token ',code);tip.append(next);}
    [...shown].sort((a,b)=>b.values[metric][i]-a.values[metric][i]).forEach(row=>{
      const line=document.createElement('div'), sw=document.createElement('i'), val=document.createElement('b'), name=document.createElement('span');
      sw.style.background=effectColor(row,nLayers); val.textContent=row.values[metric][i].toFixed(4); name.textContent=row.label+(metric==='c'?` · cumulative L2 ${formatL2(row.cumulative_length?.[i])} / ${formatL2(row.total_length)}`:'');
      line.append(sw,val,name); tip.append(line);
    });
    tip.classList.remove('hidden');
    const px=x(t[i])/w*box.width;
    tip.style.left=`${px>box.width/2?px-tip.offsetWidth-12:px+12}px`; tip.style.top='12px';
  };
  $('effect-hit').addEventListener('pointermove',move);
  $('effect-hit').addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');tip.classList.add('hidden');});
  // List
  $('effect-score-head').textContent=info.plateau_score?'Plateau score Δt':'Plateau score';
  $('effect-rows').innerHTML=effect.rows.map(row=>{
    const score=row.plateau_score[metric], checked=effectSelection.has(row.key), available=definedMetric(row,metric);
    const reason=metric==='c'?row.c_undefined_reason:row.d_undefined_reason;
    return `<tr class="${available?'':'undefined'}${checked?' selected':''}"><td><input type="checkbox" data-key="${esc(row.key)}" aria-label="Show ${esc(row.label)}"${checked?' checked':''}${available?'':' disabled'}></td>
      <th scope="row"><i class="swatch" style="background:${effectColor(row,nLayers)}"></i>${esc(row.label)}${row.patched?'<small>Patched</small>':''}</th>
      <td title="${row.endpoint_l2}">${formatL2(row.endpoint_l2)}</td>
      <td>${!available?esc(reason || 'undefined'):score==null?'—':score.toFixed(3)}</td></tr>`;
  }).join('');
}
function renderResult(record) {
  record=PlateauRecords.normalize(record); result=record; setForm(record); remember();
  renderOverview(record);
  $('inspect').classList.toggle('hidden',!record.path_predictions.length);
  $('t-slider').max=Math.max(0,record.path_predictions.length-1);
  $('t-slider').value=Math.floor(record.path_predictions.length/2);
  updateT();
  $('export-result').disabled=false;
  renderInputTokens(record.input_tokens,[record.settings.patch_start_a,record.settings.patch_start_b],record.settings.patch_position==='different_suffix');
  $('effect-section').classList.remove('hidden');
  $('effect-placeholder').classList.add('hidden');
  $('effect-metric').innerHTML=record.effect.metrics.map(m=>`<option value="${esc(m.id)}">${esc(m.label)}</option>`).join('');
  $('effect-metric').value=record.effect.rows.some(row=>row.c_status!=='not_recorded')?'c':'relative_l2_shinkle';
  applyEffectPreset(effectPreset==='custom'?'representative':effectPreset);
  const previous=$('layer-quantity').value;
  $('layer-quantity').innerHTML=layerQuantities(record).map(q=>`<option value="${esc(q.id)}">${esc(q.label)}</option>`).join('');
  if([...$('layer-quantity').options].some(o=>o.value===previous)) $('layer-quantity').value=previous;
  renderLayerPlot();
  record.predictions.forEach((prediction,i)=>{
    const side=i?'b':'a';
    $('prediction-'+side).classList.remove('muted');
    // Schema 5 records carried look-ahead tokens; only the first three are the continuation.
    const tokens=prediction.tokens.slice(0,3), ended=!!prediction.ended;
    const words=record.settings.generation==='greedy_3_words';
    $('prediction-label-'+side).textContent=words?'Next 3 words · greedy':'Next 3 tokens · greedy';
    $('prediction-'+side).innerHTML=words?`<span class="continuation">${esc(prediction.continuation)}</span>`:tokens.map((t,n)=>`<span class="word" title="ID ${t.id} · p=${(100*t.probability).toFixed(2)}%"><small>${n+1}</small>${esc(tokenText(t.text))}</span>`).join('')+
      `<span class="continuation">↳ ${esc(tokens.map(t=>t.text).join(''))}${ended?' · End of text':''}</span>`;
  });
  $('result-meta').classList.remove('hidden');
  $('result-meta').innerHTML=`<span><b>${esc(record.model_label)}</b> · ${esc(record.backend?.remote?'NDIF':(record.backend?.device || '').toUpperCase())} · ${esc(record.dtype)}</span><span>${esc(record.settings.interpolation.toUpperCase())} @ ${record.settings.patch_layer===-1?'embedding':'after layer '+record.settings.patch_layer} · ${esc(patchLabel(record.settings))} · ${record.settings.steps} samples · fixed context ${esc((record.settings.context || "a").toUpperCase())}</span><span>Logits max |Δd/Δt| <b>${Number.isFinite(record.metrics.max_abs_slope)?record.metrics.max_abs_slope.toFixed(2):'undefined'}</b> @ t ≈ ${Number.isFinite(record.metrics.peak_t)?record.metrics.peak_t.toFixed(3):'—'}</span><span>${Number.isFinite(record.elapsed_seconds)?record.elapsed_seconds.toFixed(1):'—'} s</span>`;
  $('token-details').classList.remove('hidden');
  $('token-body').innerHTML=record.predictions.map((prediction,i)=>`<div class="token-row">Generated ${i?'B':'A'}: ${prediction.tokens.slice(0,3).map(t=>`<span class="token-chip" title="ID ${t.id} · p=${(100*t.probability).toFixed(2)}%">${esc(tokenText(t.text))}</span>`).join('')}</div>`).join('')+`<p>Greedy next tokens with their probabilities (hover a chip). ␣ marks a space and ↵ a newline. Layers are numbered from 0; resid_post is recorded at the block output, before final normalization (LayerNorm for GPT-2/Pythia; RMSNorm for Qwen). Source: ${esc(record.backend?.remote?'NDIF remote':'local')} model inference · ${esc(formatDate(record.created_at, true))}.</p>`;
}
function updateT() {
  const index=Number($('t-slider').value), sample=result?.path_predictions[index];
  if(!sample)return;
  $('t-value').textContent=`t = ${sample.t.toFixed(3)}  → next token ${JSON.stringify(sample.token)}`;
  $('t-slider').setAttribute('aria-valuetext',`Sample ${index+1} of ${result.path_predictions.length}, t = ${sample.t.toFixed(3)}`);
  $('token-matrix').innerHTML=PlateauTokenMatrix.render(sample);
}
// NDIF key: per-viewer, kept in sessionStorage, or localStorage when "Remember" is checked.
const KEY_STORE='plateau-ndif-key', CODE_STORE='plateau-lab-code';  // CODE_STORE: read once to migrate
const NDIF_KEY_FORMAT=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
let requiresKey=false, sharedAccess=false;
function stored(name) { try{return localStorage.getItem(name) || sessionStorage.getItem(name) || '';}catch{return '';} }
function ndifKey() { return $('ndif-key').value.trim(); }
function saveKey() {
  const key=ndifKey(), remember=$('ndif-key-remember').checked;
  try{ key?sessionStorage.setItem(KEY_STORE,key):sessionStorage.removeItem(KEY_STORE); }catch{}
  try{ key && remember?localStorage.setItem(KEY_STORE,key):localStorage.removeItem(KEY_STORE); }catch{}
  keyState();
}
function keyState() { $('settings-toggle').classList.toggle('needs-attention',requiresKey && !ndifKey()); }
// One field: an NDIF key has a fixed format; anything else is sent as the lab access code.
function authHeaders() {
  const value=ndifKey();
  if(!value) return {};
  return NDIF_KEY_FORMAT.test(value) || !sharedAccess ? {'X-NDIF-Key':value} : {'X-Access-Code':value};
}
function showSettings(open) { $('settings-panel').classList.toggle('hidden',!open); $('settings-toggle').setAttribute('aria-expanded',String(open)); }
// History: completed runs in IndexedDB (this browser only). Every access is guarded, so the
// explorer works without it (private windows, blocked storage).
const HISTORY_DB='plateau-lab', HISTORY_STORE='runs';
let historyRuns=[];
function historyDb() {
  return new Promise((resolve,reject)=>{
    const request=indexedDB.open(HISTORY_DB,1);
    request.onupgradeneeded=()=>request.result.createObjectStore(HISTORY_STORE,{keyPath:'id'});
    request.onsuccess=()=>resolve(request.result); request.onerror=()=>reject(request.error);
  });
}
async function historyTx(mode, action) {
  const db=await historyDb();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(HISTORY_STORE,mode), store=tx.objectStore(HISTORY_STORE), request=action(store);
    tx.oncomplete=()=>{db.close();resolve(request?.result);}; tx.onerror=tx.onabort=()=>{db.close();reject(tx.error);};
  });
}
async function loadHistory() {
  let browserRuns=[];
  try { browserRuns=await historyTx('readonly',store=>store.getAll()); }
  catch { if(!localLibrary)$('history-caption').textContent='Browser storage is unavailable. Export a completed result before leaving.'; }
  const runs=new Map();
  for(const record of browserRuns) {try{runs.set(record.id,PlateauRecords.normalize(record));}catch{}}
  if(localLibrary) {
    try {
      const library=await api('/api/library'), examples=new Map(library.examples.map(r=>[r.id,r]));
      for(const record of library.history) {
        try { const merged=PlateauRecords.normalize({...record,...examples.get(record.id)}); runs.set(record.id,merged); } catch {}
      }
    } catch(error) {toast('Could not load local history: '+error.message);}
  }
  historyRuns=[...runs.values()].sort((a,b)=>b.created_at.localeCompare(a.created_at));
  for(const id of historySelection)if(!runs.has(id))historySelection.delete(id);
  $('history-count').textContent=String(historyRuns.length); renderHistory();
}
async function saveRun(record) {
  try { await historyTx('readwrite',store=>store.put(record)); await loadHistory(); }
  catch(error) { toast('Could not save this run to history: '+(error?.message || 'storage unavailable')); }
}
async function deleteRuns(ids) {
  try { if(localLibrary)await api('/api/history/delete',{ids}); await historyTx('readwrite',store=>{ids.forEach(id=>store.delete(id));}); await loadHistory(); }
  catch(error) { toast('Could not delete: '+(error?.message || 'storage unavailable')); }
}
function downloadJson(data, name) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));
  const a=document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function runFileName(record) { return `plateau-${record.model.split('/').pop()}-${record.created_at.slice(0,19).replace(/[:T]/g,'-')}.json`; }
function exportSelected() {
  const records=historyRuns.filter(r=>historySelection.has(r.id));
  if(!records.length){toast('Select at least one result.');return;}
  const format=$('history-export-format').value;
  const content=format==='csv'?PlateauRecords.csv(records):records.map(r=>JSON.stringify(r)).join('\n')+'\n';
  const url=URL.createObjectURL(new Blob([content],{type:format==='csv'?'text/csv;charset=utf-8':'application/x-ndjson'}));
  const a=document.createElement('a');a.href=url;a.download=`plateau-selected.${format}`;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderOverview(record) {
  const row=record.effect.rows.find(r=>r.key==='logits'), t=record.effect.t;
  if(!row){$('metric-overview').textContent='No logit readout in this result.';return;}
  $('metric-overview').innerHTML=['c','relative_l2_shinkle'].map(metric=>{
    const isC=metric==='c', label=isC?'Cumulative path progress c(t)':'Relative endpoint distance d(t)';
    const reason=isC?row.c_undefined_reason:row.d_undefined_reason;
    const values=row.values[metric], available=definedMetric(row,metric);
    const help=isC?'Sum of adjacent-vector L2 distances up to t, divided by the total sampled path length. The dashed c=t line means uniform progress, not necessarily a straight path.':'Distance to the A endpoint divided by the sum of distances to both endpoints. It does not measure how far the trajectory has travelled.';
    const plot=available?`<svg viewBox="0 0 720 150" role="img" aria-label="Logits ${esc(label)}"><title>Logits ${esc(label)}</title><line x1="35" y1="125" x2="690" y2="15" stroke="#bfc2ce" stroke-dasharray="4 5"/><polyline points="${values.map((v,i)=>`${35+t[i]*655},${125-v*110}`).join(' ')}" fill="none" stroke="${isC?'#7762c7':'#427eaa'}" stroke-width="2"/>${values.map((v,i)=>`<circle cx="${35+t[i]*655}" cy="${125-v*110}" r="3" fill="transparent"><title>t=${t[i]} · ${isC?'c':'d'}=${v}${isC?' · cumulative L2='+row.cumulative_length?.[i]:''}</title></circle>`).join('')}<text x="35" y="145" font-size="11">t = 0</text><text x="660" y="145" font-size="11">t = 1</text><text x="10" y="125" font-size="11">0</text><text x="10" y="20" font-size="11">1</text></svg>`:`<p>${esc(reason || 'This metric is undefined for this readout.')}</p>`;
    return `<section class="metric-card"><h2>Logits · ${label}</h2><details><summary>What does this measure?</summary><p>${help}</p></details>${isC?`<p>Total sampled path L2: <b>${formatL2(row.total_length)}</b></p>`:''}${plot}</section>`;
  }).join('');
}
async function importRuns(files) {
  let added=0, skipped=0;
  for(const file of files) {
    try {
      for(const candidate of PlateauRecords.parse(await file.text())) {
        try {const record=PlateauRecords.normalize(candidate); await historyTx('readwrite',store=>store.put(record)); added++;}
        catch {skipped++;}
      }
    } catch { skipped++; }
  }
  await loadHistory();
  toast(`Imported ${added} run${added===1?'':'s'}${skipped?` · skipped ${skipped} (unsupported or invalid result)`:''}.`);
}
function sparkline(record) {
  const row=record.effect.rows.find(r=>r.key==='logits'), metric=definedMetric(row,'c')?'c':'relative_l2_shinkle', values=row?.values[metric];
  if(!values)return '';
  const pts=values.map((v,i)=>`${(4+record.effect.t[i]*232).toFixed(1)},${(52-Math.max(-.2,Math.min(1.2,v))*44).toFixed(1)}`).join(' ');
  return `<svg viewBox="0 0 240 60" role="img" aria-label="Logits ${esc(record.effect.metrics.find(m=>m.id===metric)?.label)} along t"><line x1="4" y1="52" x2="236" y2="8" stroke="#d6d8df" stroke-dasharray="3 4"/><polyline points="${pts}" fill="none" stroke="#427eaa" stroke-width="2" stroke-linejoin="round"/></svg>`;
}
function renderHistory() {
  const query=$('history-search').value.trim().toLowerCase();
  const runs=historyRuns.filter(r=>!query || [r.sequence_a,r.sequence_b,r.model,r.model_label].some(v=>String(v).toLowerCase().includes(query)));
  $('history-clear').disabled=$('history-export-all').disabled=!historyRuns.length;
  $('history-export-selected').disabled=!historySelection.size;
  if(!runs.length){ $('history-grid').innerHTML=`<div class="library-empty">${historyRuns.length?'No runs match your search.':'No runs yet.<br>Completed experiments appear here automatically.'}</div>`; return; }
  $('history-grid').innerHTML=runs.map(r=>{
    const s=r.settings, logits=r.effect.rows.find(row=>row.key==='logits'), metric=logits && definedMetric(logits,'c')?'c':'relative_l2_shinkle', score=logits?.plateau_score[metric];
    return `<article class="example-card history-card" data-id="${esc(r.id)}">
      <div class="top"><label><input type="checkbox" data-select="${esc(r.id)}" ${historySelection.has(r.id)?'checked':''} aria-label="Select result for export"></label><span class="tag">${esc(r.model_label)}</span><span title="${esc(formatDate(r.created_at,true))}">${esc(formatDate(r.created_at,true).slice(0,16))}</span></div>
      <p><span class="prefix">A</span>${esc(r.sequence_a)}</p><p><span class="prefix">B</span>${esc(r.sequence_b)}</p>
      <p class="note">${s.patch_layer===-1?'Embedding':'After layer '+s.patch_layer} · ${esc(s.interpolation.toUpperCase())} · ${esc(patchLabel(s))} · ${s.steps} samples${score!=null?` · logits plateau score ${score.toFixed(3)}`:''}</p>
      <p class="note">${esc(r.tag || '')} ${esc(r.notes || '')}</p>
      ${sparkline(r)}
      <div class="history-actions"><button class="secondary" data-action="open">Open</button><button class="text-button" data-action="export">Export JSON</button><button class="text-button danger" data-action="delete">Delete</button></div>
    </article>`;
  }).join('');
}
function showView(view) {
  const history=view==='history';
  $('explorer').classList.toggle('hidden',history); $('history-view').classList.toggle('hidden',!history);
  $('nav-work').classList.toggle('active',!history); $('nav-history').classList.toggle('active',history);
  if(history) renderHistory();
}
async function run() {
  if(busy)return;
  if(!$('sequence-a').value.trim() || !$('sequence-b').value.trim()){status('Enter both sequences first.',0,true);return;}
  if(requiresKey && !ndifKey()){status('Add your NDIF API key (or the lab access code) in Settings (⚙) to run experiments.',0,true);showSettings(true);$('ndif-key').focus();return;}
  invalidate();setBusy(true);status('Preparing the model…');
  try { const job=await api('/api/run',settings(),authHeaders());currentJob=job.id;await poll(job.id); }
  catch(error){status('Experiment did not complete: '+error.message,0,true);}
  finally {currentJob=null;setBusy(false);}
}
async function poll(id) {
  let failures=0;
  while(true) {
    let job;
    try {job=await api('/api/jobs/'+id);failures=0;}
    catch(error){if(++failures>=5)throw new Error('Cannot connect to the server.');status('Connection interrupted. Retrying…');await new Promise(r=>setTimeout(r,1500));continue;}
    status(job.message,job.progress,job.status==='error');
    if(job.status==='done'){renderResult(job.result);saveRun(job.result);return;}
    if(job.status==='error' || job.status==='cancelled')return;
    await new Promise(r=>setTimeout(r,650));
  }
}
async function init() {
  status('Connecting to the server. A sleeping server can take up to a minute to wake…',null);
  for(let attempt=0;;attempt++){
    try { await loadConfig(); break; }
    catch(e){ if(attempt>=20){status('Cannot reach the server: '+e.message,0,true);return;} await new Promise(r=>setTimeout(r,3000)); }
  }
  $('status').classList.add('hidden');
  layers();
  try {const draft=JSON.parse(localStorage.getItem('plateau-draft-v1'));if(draft && modelLayers[draft.model]){draft.patch_position ??= 'different_suffix';setForm(draft);}}catch{}
  ['sequence-a','sequence-b'].forEach(id=>$(id).addEventListener('input',invalidate));
  ['interpolation','patch-position','steps','context'].forEach(id=>$(id).addEventListener('change',invalidate));
  $('patch-layer').addEventListener('input',invalidate);
  $('model').onchange=()=>{layers();invalidate();};
  const remembered=(()=>{try{return !!(localStorage.getItem(KEY_STORE) || localStorage.getItem(CODE_STORE));}catch{return false;}})();
  $('ndif-key').value=stored(KEY_STORE) || stored(CODE_STORE); $('ndif-key-remember').checked=remembered;
  try{ localStorage.removeItem(CODE_STORE); sessionStorage.removeItem(CODE_STORE); }catch{}
  saveKey();
  $('ndif-key').addEventListener('input',saveKey); $('ndif-key-remember').addEventListener('change',saveKey);
  $('settings-toggle').onclick=event=>{event.stopPropagation();showSettings($('settings-panel').classList.contains('hidden'));};
  document.addEventListener('click',event=>{if(!event.target.closest('.settings'))showSettings(false);});
  document.addEventListener('keydown',event=>{if(event.key==='Escape' && !$('settings-panel').classList.contains('hidden')){showSettings(false);$('settings-toggle').focus();}});
  $('effect-metric').onchange=renderEffect;
  $('effect-overlay').onchange=renderEffect;
  $('layer-quantity').onchange=renderLayerPlot;
  $('effect-preset').onchange=()=>{if($('effect-preset').value!=='custom')applyEffectPreset($('effect-preset').value);};
  $('effect-rows').addEventListener('change',event=>{
    const key=event.target.dataset?.key; if(!key)return;
    event.target.checked?effectSelection.add(key):effectSelection.delete(key);
    effectPreset='custom'; $('effect-preset').value='custom'; renderEffect();
  });
  $('nav-work').onclick=()=>showView('explorer'); $('nav-history').onclick=()=>showView('history');
  $('history-search').addEventListener('input',renderHistory);
  $('history-import').onclick=()=>$('history-file').click();
  $('history-file').onchange=()=>{importRuns([...$('history-file').files]);$('history-file').value='';};
  $('history-export-all').onclick=()=>downloadJson(historyRuns,`plateau-history-${new Date().toISOString().slice(0,10)}.json`);
  $('history-clear').onclick=()=>{if(confirm(`Delete all ${historyRuns.length} saved runs${localLibrary?' from local History and this browser':' from this browser'}? Saved examples are kept. This cannot be undone.`))deleteRuns(historyRuns.map(r=>r.id));};
  $('history-grid').addEventListener('change',event=>{const id=event.target.dataset.select;if(id){event.target.checked?historySelection.add(id):historySelection.delete(id);$('history-export-selected').disabled=!historySelection.size;}});
  $('history-export-selected').onclick=exportSelected;
  $('export-result').onclick=()=>{if(result)downloadJson(result,runFileName(result));};
  $('history-grid').addEventListener('click',event=>{
    const button=event.target.closest('button[data-action]'); if(!button)return;
    const record=historyRuns.find(r=>r.id===button.closest('[data-id]').dataset.id); if(!record)return;
    if(button.dataset.action==='open'){ if(busy){toast('Wait for the running experiment to finish.');return;} showView('explorer'); renderResult(record); $('status').classList.add('hidden'); window.scrollTo({top:0}); }
    else if(button.dataset.action==='export') downloadJson(record,runFileName(record));
    else if(button.dataset.action==='delete') deleteRuns([record.id]);
  });
  loadHistory();
  $('run').onclick=run;
  $('t-slider').oninput=updateT;
  $('cancel').onclick=async()=>{if(currentJob){try{await api(`/api/jobs/${currentJob}/cancel`,{});toast('Stop requested. The current request must finish first.');}catch(e){toast(e.message);}}};
  $('swap').onclick=()=>{const a=$('sequence-a').value;$('sequence-a').value=$('sequence-b').value;$('sequence-b').value=a;invalidate();};
  document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>preset(Number(b.dataset.preset)));
  $('next-preset').onclick=()=>{preset(presetIndex);presetIndex=(presetIndex+1)%presets.length;};
  window.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key==='Enter'){event.preventDefault();run();}});
  invalidate();
}
init();
