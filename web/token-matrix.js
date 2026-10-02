/* Shared by the Explorer and classic workbench. Pure saved-result rendering. */
'use strict';
const PlateauTokenMatrix = (() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const piece = value => String(value).replace(/ /g,'␣').replace(/\n/g,'↵').replace(/\t/g,'⇥').replace(/\r/g,'␍') || '∅';
  function render(sample) {
    const matrix=sample?.token_matrix;
    if(!matrix || !Array.isArray(matrix.steps)) return '<p class="token-matrix-notice">Top-3 predictions were not recorded for this result. Rerun the experiment to inspect the next three tokens.</p>';
    const steps=matrix.steps.slice(0,3), prefix=[];
    const reason=matrix.stop_reason==='eos'?'Generation ended at the end-of-text token.':matrix.stop_reason==='nonfinite'?'Prediction unavailable: the model returned nonfinite probabilities.':'';
    const headers=Array.from({length:3},(_,step)=>{
      const after=step===0?'At the patched sample':steps[step]?`After ${JSON.stringify(piece(prefix.join('')))}`:reason || 'Not recorded';
      if(steps[step]?.[0])prefix.push(steps[step][0].text);
      return `<th scope="col">Token +${step+1}<small>${escape(after)}</small></th>`;
    }).join('');
    const rows=Array.from({length:3},(_,rank)=>`<tr${rank===0?' class="token-matrix-greedy"':''}><th scope="row">${rank+1}${rank===0?'<small>Greedy</small>':''}</th>${Array.from({length:3},(_,step)=>{
      const token=steps[step]?.[rank];
      if(!token)return `<td class="token-matrix-empty"><span aria-label="${escape(reason || 'Candidate not recorded')}">—</span></td>`;
      const valid=Number.isFinite(token.probability) && token.probability>=0 && token.probability<=1;
      const percent=valid?token.probability*100:0;
      const label=valid?(percent>0 && percent<.01?'<0.01':percent.toFixed(2))+'%':'Unavailable';
      return `<td><code title="Token ID ${escape(token.id)}">${escape(piece(token.text))}</code>${token.is_eos?'<span class="token-matrix-eos">End of text</span>':''}<span class="token-matrix-prob">${escape(label)}</span><span class="token-matrix-bar" aria-hidden="true"><i style="width:${percent}%"></i></span></td>`;
    }).join('')}</tr>`).join('');
    return `<div class="token-matrix-heading"><strong>Next 3 tokens · top 3 candidates</strong><span>Greedy continuation: <code>${escape(piece(prefix.join('')))}</code></span></div><div class="token-matrix-scroll" tabindex="0" role="region" aria-label="Top three candidates for the next three token positions"><table class="token-matrix-table"><caption class="sr-only">Rows are probability ranks. Each later column follows the previous columns’ rank-one tokens.</caption><thead><tr><th scope="col">Rank</th>${headers}</tr></thead><tbody>${rows}</tbody></table></div><p class="token-matrix-note">Each column follows the earlier rank-one choices. Probabilities use the full vocabulary; the three shown need not sum to 100%. ␣ = space · ↵ = newline. Tokens can be word fragments.${reason?' '+escape(reason):''}</p>`;
  }
  return {render};
})();
if(typeof module!=='undefined' && module.exports)module.exports=PlateauTokenMatrix;
