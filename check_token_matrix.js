// Pure shared renderer: conditional columns, end-of-text, escaping and legacy data.
'use strict';
const assert=require('node:assert/strict');
const {render}=require('./web/token-matrix.js');
const records=require('./web/records.js');
const candidate=(id,text,probability,is_eos=false)=>({id,text,probability,is_eos});
const sample={t:.5,token_id:1,token:' a',token_matrix:{steps:[
  [candidate(1,' a',.4),candidate(2,'<script>',.3),candidate(3,'\n',.1)],
  [candidate(4,' b',.2),candidate(5,'"quoted"',.1),candidate(6,' c',.000001)],
  [candidate(7,' d',.6),candidate(8,' e',.2),candidate(9,' f',.1)]],stop_reason:null}};
const html=render(sample);
assert.equal((html.match(/<td>/g)||[]).length,9);
assert.match(html,/After &quot;␣a&quot;/);
assert.match(html,/After &quot;␣a␣b&quot;/);
assert.match(html,/40.00%/);
assert.match(html,/&lt;0.01%/);
assert.match(html,/&lt;script&gt;/);
assert(!html.includes('<script>'));
assert.match(html,/↵/);
assert.match(html,/full vocabulary/);
assert.match(render({t:0,token:'old'}),/Rerun the experiment/);
const eos=JSON.parse(JSON.stringify(sample));
eos.token_matrix.steps=eos.token_matrix.steps.slice(0,1);
eos.token_matrix.steps[0][0]=candidate(0,'<eos>',.5,true);
eos.token_matrix.stop_reason='eos';
const ended=render(eos);
assert.match(ended,/Generation ended/);
assert.equal((ended.match(/token-matrix-empty/g)||[]).length,6);
assert(!ended.includes('NaN'));
assert.match(render({token_matrix:{steps:[],stop_reason:'nonfinite'}}),/nonfinite probabilities/);
const fixture=require('./docs/assets/measured-example.json');
fixture.id ??= 'a'.repeat(32);
fixture.created_at ??= '2026-10-01T00:00:00Z';
fixture.path_predictions ??= [];
fixture.path_predictions[0]=sample;
const normalized=records.normalize(records.parse(JSON.stringify(fixture)+'\n')[0]);
assert.deepEqual(normalized.path_predictions[0],sample);
console.log('PASS: 3 × 3 conditional matrix, full probabilities, EOS, nonfinite, escaping, legacy notice and JSONL preservation.');
