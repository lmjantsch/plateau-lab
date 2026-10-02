// Real migration/render/import functions with DOM/storage stubs; not browser QA.
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const records=require('./web/records.js');
const fixture=JSON.parse(fs.readFileSync(process.argv[2] || 'docs/assets/measured-example.json','utf8'));
fixture.id ??= 'a'.repeat(32); fixture.created_at ??= '2026-09-30T00:00:00Z';
fixture.notes='Unicode α\n"quote", comma'; fixture.tag='Saved';
const current=records.normalize(fixture);
assert.deepEqual(current.curves,fixture.curves);
assert.equal(current.notes,fixture.notes);
const legacy=JSON.parse(JSON.stringify(fixture)); legacy.schema_version=3;delete legacy.effect;
legacy.curves=legacy.curves.map(({key,title,t,d})=>({key,title,t,d}));
const old=records.normalize(legacy);
assert(old.effect.rows.every(row=>row.values.c===null && row.c_status==='not_recorded'));
for(const schema of [1,2,3,4,5,6,7])assert.equal(records.normalize({...current,schema_version:schema}).schema_version,schema);
assert.throws(()=>records.normalize({...current,schema_version:99}));
assert.equal(records.parse(JSON.stringify(fixture)+'\n'+JSON.stringify(legacy)).length,2);
const csv=records.csv([current,old]);
assert(csv.includes('"Unicode α\n""quote"", comma"'));
assert(csv.includes('"cumulative_length"') && csv.includes('"not_recorded"'));

const elements=new Map();
function element(id) {
  const classes=new Set(),handlers={};
  return {id,handlers,innerHTML:'',textContent:'',value:'',checked:false,style:{},
    get options(){return [...this.innerHTML.matchAll(/<option value="([^"]*)"/g)].map(m=>({value:m[1]}));},
    classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c),toggle:(c,on)=>on?classes.add(c):classes.delete(c)},
    setAttribute(){},addEventListener:(name,fn)=>{handlers[name]=fn;},querySelector:()=>element('svg'),
    getBoundingClientRect:()=>({left:0,width:720}),replaceChildren(){},append(){}};
}
const html=fs.readFileSync('web/index.html','utf8');
for(const m of html.matchAll(/id="([^"]+)"/g)){assert(!elements.has(m[1]),'duplicate id '+m[1]);elements.set(m[1],element(m[1]));}
const document={getElementById:id=>{if(!elements.has(id))elements.set(id,element(id));return elements.get(id);},
  querySelectorAll:()=>[],createElement:tag=>element(tag)};
const context=vm.createContext({PlateauRecords:records,PlateauTokenMatrix:require('./web/token-matrix.js'),document,window:{},console,
  localStorage:{setItem(){}},fixture:current,legacy:old,saved:[],messages:[]});
vm.runInContext(fs.readFileSync('web/app.js','utf8').replace(/\ninit\(\);\s*$/,''),context);
const run=code=>vm.runInContext(code,context);
run('modelLayers[fixture.model]=6; renderResult(fixture)');
assert.equal(elements.get('effect-metric').value,'c');
assert.match(elements.get('metric-overview').innerHTML,/Cumulative path progress c\(t\)/);
assert.match(elements.get('metric-overview').innerHTML,/Relative endpoint distance d\(t\)/);
run(`result.path_predictions=[{t:0,token:' <first>',token_matrix:{steps:[[{id:1,text:' <first>',probability:.6}], [{id:2,text:' next',probability:.2}], [{id:3,text:' last',probability:.1}]],stop_reason:null}},{t:1,token:'old'}];$('t-slider').value=0;updateT();`);
assert.match(elements.get('token-matrix').innerHTML,/␣&lt;first&gt;/);
assert.match(elements.get('token-matrix').innerHTML,/60.00%/);
run(`delete result.path_predictions[1].token_matrix;$('t-slider').value=1;updateT();`);
assert.match(elements.get('token-matrix').innerHTML,/Rerun the experiment/);
assert(!elements.get('token-matrix').innerHTML.includes('&lt;first&gt;'));
run('renderResult(legacy)');
assert.equal(elements.get('effect-metric').value,'relative_l2_shinkle');
assert.match(elements.get('metric-overview').innerHTML,/rerun to measure/);
assert.match(run('sparkline(legacy)'),/Relative endpoint distance/);
// An undefined d must never disable a valid c readout.
run(`var loop=JSON.parse(JSON.stringify(fixture));loop.effect.rows.forEach(row=>{
  row.values.relative_l2_shinkle=null;row.d_status='coincident_endpoints';row.d_undefined_reason='Coincident endpoints';
});renderResult(loop);`);
assert.match(elements.get('effect-plot').innerHTML,/<polyline/);
assert.match(elements.get('metric-overview').innerHTML,/Coincident endpoints/);
run(`historyTx=async(mode,action)=>action({put:record=>saved.push(record)});loadHistory=async()=>{};toast=m=>messages.push(m);`);
(async()=>{
  await run('importRuns([{text:async()=>JSON.stringify(fixture)+"\\n"+JSON.stringify(legacy)}])');
  assert.equal(context.saved.length,2);assert.match(context.messages[0],/Imported 2 runs/);
  console.log('PASS: schemas 1–7, legacy d-only, c/d rendering, independent validity, actual JSONL import, notes and CSV quoting. DOM stub, not browser validation.');
})().catch(error=>{console.error(error);process.exitCode=1;});
