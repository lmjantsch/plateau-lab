// Component/markup checks with a small DOM stub. This is NOT browser validation.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname,'static/index.html'),'utf8');
const elements = new Map();
function element(id) {
  const classes=new Set(),handlers={},attrs={};
  return {id,handlers,attrs,innerHTML:'',textContent:'',value:'0',
    classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c),
      toggle:(c,on)=>{on?classes.add(c):classes.delete(c);}},
    style:{setProperty:(key,value)=>{attrs[key]=value;}},
    setAttribute:(key,value)=>{attrs[key]=value;},
    addEventListener:(event,fn)=>{handlers[event]=fn;},
    closest:()=>elements.get('help-group'),
    contains:target=>target===elements.get('c-info') || target===elements.get('c-definition')};
}
for(const match of html.matchAll(/id="([^"]+)"/g)) {
  assert(!elements.has(match[1]),`duplicate id: ${match[1]}`);
  elements.set(match[1],element(match[1]));
}
elements.set('help-group',element('help-group'));
const document={activeElement:null,handlers:{},getElementById:id=>{
  assert(elements.has(id),`missing element: ${id}`);return elements.get(id);
},addEventListener:(name,fn)=>{document.handlers[name]=fn;}};
const context=vm.createContext({document,console,PlateauTokenMatrix:require('./web/token-matrix.js')});
const source=fs.readFileSync(path.join(__dirname,'static/app.js'),'utf8').replace(/\ninit\(\);\s*$/,'');
vm.runInContext(source,context);
function run(code){return vm.runInContext(code,context);}
run(`var curve={key:'logits',title:'Logits',t:[0,.5,1],d:[0,.2,1],c:[0,.75,1],
  cumulative_length:[0,3,4],total_length:4}; var record={curves:[curve],metrics:{c:{max_abs_slope:1.5,peak_t:.25},d:{max_abs_slope:1.6,peak_t:.75}}};`);
assert.match(run('curveSvg(curve)'),/cumulative path progress c\(t\)/);
assert.match(run('curveSvg(curve)'),/c\(t\)=0.75000 · cumulative L2=3/);
assert.match(run('curveSvg(curve,false,"d")'),/d\(t\)=0.20000/);
assert.match(run('curveSvg(curve)'),/Uniform path progress/);
run('renderCurves(record,"c");renderCurves(record,"d");');
assert.match(elements.get('c-charts').innerHTML,/Total path L2 · 4/);
assert.match(elements.get('charts').innerHTML,/relative endpoint distance d\(t\)/);
assert(html.indexOf('id="c-section"')<html.indexOf('id="d-section"'));
assert.match(run('metricSummary(record,"c")'),/Δc\/Δt/);
assert.match(run('metricSummary(record,"d")'),/Δd\/Δt/);
run('result={...record,path_predictions:[{t:0,token:"A"},{t:.5,token:"B"},{t:1,token:"C"}]};');
elements.get('t-slider').value='1';run('updateT()');
assert.match(elements.get('token-matrix').innerHTML,/Rerun the experiment/);
run(`result.path_predictions[1].token_matrix={steps:[[{id:1,text:' <one>',probability:.5}], [{id:2,text:' two',probability:.25}], [{id:3,text:' three',probability:.1}]],stop_reason:null};updateT();`);
assert.match(elements.get('token-matrix').innerHTML,/␣&lt;one&gt;/);
assert.match(elements.get('token-matrix').innerHTML,/50.00%/);
assert.match(elements.get('token-matrix').innerHTML,/Token \+3/);
elements.get('t-slider').value='2';run('updateT()');
assert.match(elements.get('token-matrix').innerHTML,/Rerun the experiment/);
elements.get('t-slider').value='1';run('updateT()');
assert.match(elements.get('sample-values').innerHTML,/c\(t\): 0.75000/);
assert.match(elements.get('sample-values').innerHTML,/d\(t\): 0.20000/);
assert.match(elements.get('sample-values').innerHTML,/Cumulative L2: 3 \/ total path L2: 4/);
run('var legacy={key:"logits",title:"Logits",t:[0,.5,1],d:[0,.2,1]};');
assert.match(run('libraryPreview({curves:[legacy]})'),/Legacy · d\(t\) only/);
assert.match(run('curveSvg(legacy)'),/requires rerunning/);
assert(!run('curveSvg(legacy)').includes('<svg'));
assert.match(run('libraryPreview(record)'),/c\(t\) · Cumulative path progress/);
run('var stationary={...curve,c:[null,null,null],d:[null,null,null],c_undefined_reason:"No measured movement; c(t) undefined",d_undefined_reason:"Coincident endpoints; d(t) undefined"};');
assert.match(run('curveSvg(stationary)'),/No measured movement; c\(t\) undefined/);
assert(!run('curveSvg(stationary)').includes('<polyline'));
assert.match(run('curveSvg({...stationary,c:curve.c})'),/<polyline/);
assert(!run('curveSvg({...stationary,c:curve.c},false,"d")').includes('<polyline'));
assert.match(run('metricSummary({metrics:{c:{max_abs_slope:null}}},"c")'),/undefined/);
// Exercise help event handlers separately; actual focus/hover/layout needs a browser.
run('setupMetricHelp()');
const button=elements.get('c-info'),help=elements.get('c-definition'),group=elements.get('help-group');
group.handlers.mouseenter();assert.equal(button.attrs['aria-expanded'],'true');
group.handlers.mouseleave();assert(help.classList.contains('hidden'));
document.activeElement=button;button.handlers.focus();assert(!help.classList.contains('hidden'));
button.handlers.keydown({key:'Escape'});assert(help.classList.contains('hidden'));
button.handlers.click();assert(!help.classList.contains('hidden'));
button.handlers.click();assert(help.classList.contains('hidden'));
document.handlers.pointerdown({target:{}});assert.equal(button.attrs['aria-expanded'],'false');
if(process.argv[2]) {
  context.fixture=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
  run('modelLayers[fixture.model]=6; renderResult(fixture);');
  assert.match(elements.get('c-charts').innerHTML,/cumulative path progress/);
  assert.match(elements.get('charts').innerHTML,/relative endpoint distance/);
  assert(elements.get('c-legacy').classList.contains('hidden'));
  assert.match(elements.get('result-meta').innerHTML,/Δc\/Δt/);
  run(`var oldFixture=JSON.parse(JSON.stringify(fixture));
    oldFixture.schema_version=3;delete oldFixture.metrics.c;delete oldFixture.metrics.d;
    oldFixture.curves=oldFixture.curves.map(({key,title,t,d})=>({key,title,t,d}));
    renderResult(oldFixture);`);
  assert(!elements.get('c-legacy').classList.contains('hidden'));
  assert.match(elements.get('c-charts').innerHTML,/requires rerunning/);
  assert.match(elements.get('charts').innerHTML,/<polyline/);
  assert.match(elements.get('result-meta').innerHTML,/Logits c\(t\): rerun to measure/);
  assert(!elements.get('sample-values').innerHTML.includes('NaN'));
  console.log('PASS: full result restoration from real-model and legacy fixtures (DOM stub, not a browser).');
}
console.log('PASS: component markup, c/d labels and values, legacy/undefined rendering, summaries, sample inspector, and help handlers. Browser layout/interactions not tested.');
