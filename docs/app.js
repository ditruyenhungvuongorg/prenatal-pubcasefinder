'use strict';
const $ = id => document.getElementById(id);
const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let base = window.PRENATAL_CONFIG?.apiBase || '';
try { base = localStorage.getItem('prenatal-api') || base; } catch {}
const selection = new CaseSelection(), chosen = selection.items, labIds = new Set(), requests = new Set();
let access = '', result = null, revision = 0, searchVersion = 0, extractVersion = 0, busy = false;
let searchTerms = [], extraction = null, connectionVersion = 0, timer;
function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function localStatus(id, message = '') { $(id).textContent = message; }
async function api(path, payload, timeout = 120000, controller = new AbortController()) {
  if (!base && location.hostname.endsWith('github.io')) throw Error('Mở Kết nối để nhập địa chỉ máy chủ Ubuntu.');
  requests.add(controller);
  const timeoutId = setTimeout(() => controller.abort('timeout'), timeout);
  try {
    const response = await fetch(base + path, {
      method: payload ? 'POST' : 'GET',
      headers: {...(payload ? {'Content-Type':'application/json'} : {}), ...(access ? {Authorization:'Bearer ' + access} : {})},
      ...(payload ? {body:JSON.stringify(payload)} : {}), signal:controller.signal, cache:'no-store'
    });
    let data;
    try { data = await response.json(); } catch { throw Error('Máy chủ trả dữ liệu không hợp lệ. Kiểm tra lại kết nối.'); }
    if (!response.ok) throw Error(data.error || 'Không xử lý được yêu cầu. Vui lòng thử lại.');
    return data;
  } catch (e) {
    if (controller.signal.aborted) throw Error(controller.signal.reason === 'timeout' ? 'Máy chủ phản hồi chậm. Danh sách HPO vẫn được giữ; bạn có thể thử lại.' : 'Yêu cầu đã hủy.');
    if (e instanceof TypeError) throw Error('Chưa kết nối được máy chủ. Kiểm tra Ubuntu và mã truy cập; danh sách HPO vẫn được giữ.');
    throw e;
  } finally { clearTimeout(timeoutId); requests.delete(controller); }
}
function updateActions() {
  $('reviewed').disabled = busy || !!selection.conflicts || !chosen.size;
  $('match').disabled = busy || !!selection.conflicts || !chosen.size || !$('reviewed').checked;
  $('extract').disabled = busy;
  $('extract').textContent = busy === 'extract' ? 'Đang tách dấu hiệu…' : 'Tách dấu hiệu từ văn bản';
  $('match').textContent = busy === 'match' ? 'Đang đối chiếu…' : 'Đối chiếu bệnh →';
  $('export').disabled = !result || !labIds.size || !!selection.conflicts || !!busy;
  $('conflict-summary').hidden = !selection.conflicts;
  $('conflict-summary').textContent = selection.conflicts ? selection.conflicts + ' HPO có trạng thái khác nhau giữa các nguồn. Chọn trạng thái cuối cùng trước khi đối chiếu.' : '';
}
function invalidate() {
  revision++; result = null; labIds.clear(); $('reviewed').checked = false; $('pattern').hidden = true;
  $('results-title').textContent = 'Kết quả bệnh tiềm năng';
  $('inheritance-filter').innerHTML = '<option value="">Tất cả kiểu di truyền</option>';
  $('results').innerHTML = '<div class="empty"><div class="empty-symbol">↻</div><h3>Sẵn sàng cho lần đối chiếu tiếp theo</h3><p>Duyệt danh sách HPO hiện tại để xem kết quả phù hợp với ca này.</p></div>';
  if ($('lab').open) $('lab').close();
  updateActions();
}
function add(term, state = 'CÓ', source = {kind:'manual'}) {
  const outcome = selection.add(term, state, source);
  if (outcome !== 'duplicate') invalidate();
  notice(outcome === 'conflict'
    ? 'HPO ' + term.id + ' có trạng thái mâu thuẫn. Trạng thái cũ được giữ cho đến khi bác sĩ quyết định.'
    : outcome === 'duplicate' ? 'HPO ' + term.id + ' đã có; giữ nguyên trạng thái bác sĩ đã chọn.'
    : 'Đã thêm ' + (term.vi || term.en) + ' vào danh sách HPO của ca.');
  renderSelected(); renderSearch(); renderMentions();
  return outcome;
}
function renderSelected() {
  $('hpo-count').textContent = chosen.size;
  $('selected').innerHTML = chosen.size ? '' : '<div class="selection-empty">Chưa có dấu hiệu nào được chọn. Tìm HPO hoặc tách đoạn văn ở hàng phía trên.</div>';
  for (const h of chosen.values()) {
    const div = document.createElement('div'); div.className = 'chosen' + (h.pending.length ? ' has-conflict' : '');
    const sourceNames = [...new Set(h.sources.map(s => s.kind === 'text' ? 'Đoạn văn' : s.kind === 'feature' ? 'Khảo sát bổ sung' : 'Tìm HPO'))];
    const stale = selection.stale(h.id, $('clinical-text').value);
    div.innerHTML = '<div class="chosen-head"><strong>' + escape(h.vi || h.en) + '</strong><small>' + escape(h.en) + ' · ' + escape(h.id) + '</small></div>' +
      '<div class="source-label">' + sourceNames.map(s => '<span>' + escape(s) + '</span>').join('') + (stale ? '<small class="stale-source">Đoạn văn đã thay đổi · cần duyệt lại</small>' : '') + '</div>' +
      '<label class="status-control"><span>Trạng thái</span><select aria-label="Trạng thái ' + h.id + '">' + ['CÓ','NGHI NGỜ','KHÔNG'].map(s => '<option ' + (s === h.status ? 'selected' : '') + '>' + s + '</option>').join('') + '</select></label>' +
      '<button class="remove-hpo quiet" aria-label="Xóa ' + h.id + '">Xóa</button>';
    div.querySelector('.remove-hpo').onclick = () => { selection.remove(h.id); invalidate(); renderSelected(); renderSearch(); renderMentions(); };
    div.querySelector('select').onchange = e => { selection.resolve(h.id, e.target.value); invalidate(); renderSelected(); renderSearch(); renderMentions(); notice('Đã cập nhật trạng thái. Duyệt danh sách trước khi đối chiếu lại.'); };
    if (h.pending.length) {
      const conflict = document.createElement('div'); conflict.className = 'conflict';
      const states = [...new Set(h.pending.map(p => p.status))];
      conflict.innerHTML = '<strong>Cần xác nhận trạng thái</strong><p>Đang chọn: ' + escape(h.status) + '. Nguồn mới gợi ý: ' + states.map(escape).join(', ') + '.</p>';
      for (const state of [h.status, ...states.filter(s => s !== h.status)]) {
        const b = document.createElement('button'); b.className = 'quiet'; b.textContent = (state === h.status ? 'Giữ ' : 'Chọn ') + state;
        b.onclick = () => { selection.resolve(h.id, state); invalidate(); renderSelected(); renderSearch(); renderMentions(); notice('Đã xác nhận trạng thái ' + state + ' cho ' + h.id + '.'); };
        conflict.append(b);
      }
      div.append(conflict);
    }
    $('selected').append(div);
  }
  updateActions();
}
function renderSearch() {
  $('search-results').replaceChildren();
  for (const term of searchTerms) {
    const h = chosen.get(term.id), b = document.createElement('button');
    b.className = 'search-item' + (h ? ' is-selected' : '');
    b.innerHTML = '<span><strong>' + escape(term.vi || term.en) + '</strong><small>' + escape(term.en) + ' · ' + escape(term.id) + '</small></span><span class="selection-action">' + (h ? 'Đã chọn · ' + escape(h.status) : '+ Thêm HPO') + '</span>';
    b.onclick = () => add(term);
    $('search-results').append(b);
  }
}
function renderMentions() {
  $('mentions').replaceChildren();
  if (!extraction || extraction.text !== $('clinical-text').value) return;
  for (const m of extraction.mentions) {
    const box = document.createElement('div'); box.className = 'mention';
    box.innerHTML = '<div class="mention-heading"><blockquote>“' + escape(m.mention_text) + '”</blockquote><span class="subtle">Cụm nguyên văn</span></div>' +
      '<p class="hint">' + escape(m.explanation) + (m.context_review ? ' · Kiểm tra phủ định, chủ thể và tiền sử trong câu gốc.' : '') + '</p>';
    const status = document.createElement('select'); status.setAttribute('aria-label', 'Trạng thái gợi ý: ' + m.mention_text);
    status.innerHTML = (m.context_review ? '<option value="">Bác sĩ chọn trạng thái…</option>' : '') + ['CÓ','NGHI NGỜ','KHÔNG'].map(s => '<option>' + s + '</option>').join('');
    status.value = m.reviewStatus; status.onchange = () => { m.reviewStatus = status.value; renderMentions(); };
    const label = document.createElement('label'); label.className = 'mention-status'; label.textContent = 'Trạng thái trước khi thêm'; label.append(status); box.append(label);
    if (!m.candidates.length) {
      const p = document.createElement('p'); p.className = 'hint'; p.textContent = 'Chưa tìm thấy mã phù hợp. Tìm HPO bằng một cụm ngắn hơn.'; box.append(p);
    }
    for (const term of m.candidates) {
      const h = chosen.get(term.id), b = document.createElement('button'); b.className = 'search-item' + (h ? ' is-selected' : '');
      const different = h && m.reviewStatus && h.status !== m.reviewStatus;
      b.innerHTML = '<span><strong>' + escape(term.vi || term.en) + '</strong><small>' + escape(term.id) + ' · ' + escape(term.en) + '</small></span><span class="selection-action">' + (different ? 'Khác trạng thái · xem xét' : h ? 'Đã chọn · ' + escape(h.status) : '+ Thêm HPO') + '</span>';
      b.disabled = !m.reviewStatus;
      b.onclick = () => add(term, m.reviewStatus, {kind:'text', text:extraction.text, phrase:m.mention_text, start:m.span_start});
      box.append(b);
    }
    $('mentions').append(box);
  }
}
function activateTab(mode, focus = true) {
  $('manual').hidden = mode !== 'search'; $('paragraph').hidden = mode !== 'text';
  for (const m of ['search','text']) { $('tab-' + m).setAttribute('aria-selected', String(m === mode)); $('tab-' + m).tabIndex = m === mode ? 0 : -1; }
  if (focus) $(mode === 'search' ? 'query' : 'clinical-text').focus();
}
for (const mode of ['search','text']) {
  $('tab-' + mode).onclick = () => activateTab(mode);
  $('tab-' + mode).onkeydown = e => { if (['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) {e.preventDefault(); const next = e.key === 'Home' ? 'search' : e.key === 'End' ? 'text' : mode === 'search' ? 'text' : 'search'; activateTab(next, false); $('tab-' + next).focus();} };
}
let searchController;
$('query').oninput = () => {
  clearTimeout(timer); searchController?.abort('superseded');
  const seq = ++searchVersion, q = $('query').value.trim(); searchTerms = []; renderSearch();
  localStatus('search-status', q ? 'Đang tìm thuật ngữ…' : 'Nhập tên dấu hiệu để bắt đầu tìm.');
  if (!q) return;
  timer = setTimeout(async () => {
    searchController = new AbortController();
    try {
      const terms = await api('/api/hpo_search?q=' + encodeURIComponent(q), null, 20000, searchController);
      if (seq !== searchVersion) return;
      searchTerms = terms; renderSearch(); localStatus('search-status', terms.length ? terms.length + ' gợi ý · Chọn để thêm vào danh sách chung.' : 'Không tìm thấy. Thử cụm ngắn hơn hoặc mã HPO.');
    } catch (e) { if (seq === searchVersion) localStatus('search-status', e.message); }
  },250);
};
$('clinical-text').oninput = () => {
  extractVersion++; extraction = null; renderMentions(); invalidate(); renderSelected();
  localStatus('extract-status', 'Đoạn văn đã thay đổi. HPO đã chọn vẫn được giữ; tách lại để cập nhật gợi ý.');
};
$('extract').onclick = async () => {
  if (busy) return;
  const text = $('clinical-text').value;
  if (!text.trim()) { localStatus('extract-status','Nhập đoạn mô tả siêu âm trước.'); return; }
  const seq = ++extractVersion; busy = 'extract'; updateActions();
  localStatus('extract-status','Model 1 đang tách dấu hiệu. Các HPO đã chọn được giữ nguyên.');
  try {
    const data = await api('/api/extract_hpo', {text});
    if (seq !== extractVersion) return;
    extraction = {text, mentions:data.mentions.map(m => ({...m, reviewStatus:m.context_review ? '' : m.status}))};
    renderMentions();
    localStatus('extract-status', data.mentions.length ? 'Tìm thấy ' + data.mentions.length + ' cụm. Chọn HPO phù hợp bên dưới; không tự ghi đè danh sách đã duyệt.' : 'Chưa tách được dấu hiệu. Bạn vẫn có thể tìm HPO thủ công.');
  } catch (e) { if (seq === extractVersion) localStatus('extract-status', e.message); }
  finally { busy = false; updateActions(); }
};
$('clear-hpos').onclick = () => { selection.clear(); invalidate(); renderSelected(); renderSearch(); renderMentions(); notice('Đã xóa danh sách HPO. Đoạn văn và gợi ý vẫn được giữ.'); };
$('reviewed').onchange = updateActions;
$('match').onclick = async () => {
  if (busy || selection.conflicts || !$('reviewed').checked) return;
  const seq = revision;
  const snapshot = {hpos:structuredClone([...chosen.values()]), text:$('clinical-text').value};
  busy = 'match'; updateActions(); notice('Đang đối chiếu hồ sơ bệnh hiếm…');
  try {
    const data = await api('/api/match_diseases', {hpos:snapshot.hpos.map(h => ({id:h.id,status:h.status}))});
    if (seq !== revision) return;
    result = {...data, ...snapshot};
    labIds.clear(); for (const c of data.candidates.slice(0,5)) labIds.add(c.disease_id);
    const modes = [...new Set(data.candidates.flatMap(c => c.inheritance_modes))].sort();
    $('inheritance-filter').innerHTML = '<option value="">Tất cả kiểu di truyền</option>' + modes.map(m => '<option>' + escape(m) + '</option>').join('');
    renderResults(); notice('');
  } catch (e) { if (seq === revision) {result = null;labIds.clear();notice(e.message);} }
  finally { busy = false; updateActions(); }
};

function pills(values,cls){return values.length?values.map(v=>'<span class="pill '+cls+'">'+escape(v)+'</span>').join(''):'<span class="hint">Chưa có thông tin</span>';}
function renderResults(){
  if(!result)return;
  const filter=$('inheritance-filter').value;
  const list=result.candidates.filter(c=>!filter||c.inheritance_modes.includes(filter));
  $('results-title').textContent=list.length+' bệnh tiềm năng';
  $('pattern').textContent=result.clinical_pattern;$('pattern').hidden=!result.clinical_pattern;
  $('results').replaceChildren();
  if(!list.length)$('results').innerHTML='<p class="no-results">Không có kết quả phù hợp bộ lọc.</p>';
  for(const c of list){
    const card=document.createElement('article');card.className='disease';
    const match=c.disease_id.match(/^(OMIM|ORPHA|ORPHANET):(\d+)$/);
    const link=match?(match[1]==='OMIM'?'https://omim.org/entry/':'https://www.orpha.net/en/disease/detail/')+match[2]:'';
    card.innerHTML='<div class="disease-top"><div class="rank">'+c.rank+'</div><div class="disease-title"><h3>'+escape(c.disease_name)+'</h3>'+(link?'<a target="_blank" rel="noopener noreferrer" href="'+link+'">'+escape(c.disease_id)+' ↗</a>':escape(c.disease_id))+'</div><div class="score">'+c.match_percentage+'%<small>TƯƠNG ĐỒNG</small></div></div>'+
      '<div class="evidence-row"><span class="evidence-label">HPO khớp</span><div>'+pills(c.matched_phenotypes.map(p=>(p.vi||p.en)+' · '+p.id+(p.relation==='exact'?'':' (ngữ nghĩa)')),'blue')+'</div></div>'+
      '<div class="evidence-row"><span class="evidence-label">Kiểu di truyền</span><div>'+pills(c.inheritance_modes,'green')+'</div></div>'+
      '<div class="evidence-row"><span class="evidence-label">Gen liên quan</span><div>'+pills(c.causative_genes,'gray')+'</div></div>'+
      '<details class="features"><summary>Dấu hiệu chưa ghi nhận · Xem '+c.clinical_features_to_check.length+'</summary><p class="hint">Bác sĩ xem xét tính phù hợp với tuổi thai và khả năng khảo sát. Danh sách có thể gồm dấu hiệu sau sinh; chưa ghi nhận không có nghĩa là âm tính.</p><div class="feature-list"></div></details>'+
      (c.model3_rationale||c.model3_recommended_tests?'<div class="recommendation"><strong>Gợi ý hội chẩn · cần bác sĩ duyệt</strong><p>'+escape(c.model3_rationale)+'</p>'+escape(c.model3_recommended_tests)+'</div>':'')+
      '<label class="check lab-choice"><input type="checkbox" '+(labIds.has(c.disease_id)?'checked':'')+'> Đưa bệnh này vào phiếu Lab</label>';
    const featureList=card.querySelector('.feature-list');
    for(const t of c.clinical_features_to_check){
      const b=document.createElement('button');b.className='pill amber';b.textContent=(t.vi||t.en)+' · '+t.id+' +';
      b.title='Thêm vào ca với trạng thái Nghi ngờ để bác sĩ duyệt';
      b.onclick=()=>add(t,'NGHI NGỜ',{kind:'feature'});
      featureList.append(b);
    }
    card.querySelector('input').onchange=e=>{if(e.target.checked)labIds.add(c.disease_id);else labIds.delete(c.disease_id);updateActions();};
    $('results').append(card);
  }
}
$('inheritance-filter').onchange=renderResults;
$('export').onclick=()=>{
  if(!result||!labIds.size)return;
  const rows=result.candidates.filter(c=>labIds.has(c.disease_id));
  const table=(heads,body)=>'<table><thead><tr>'+heads.map(h=>'<th>'+escape(h)+'</th>').join('')+'</tr></thead><tbody>'+body.map(r=>'<tr>'+r.map(v=>'<td>'+escape(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  $('sheet').innerHTML='<p style="text-align:center">HỘI CHẨN DI TRUYỀN TIỀN SẢN</p><h1>PHIẾU GỬI LAB DI TRUYỀN</h1><p>Mã ca: '+escape($('case-id').value||'……………………')+' · Tuổi thai: '+escape($('gestation').value||'……………………')+'\nBác sĩ: '+escape($('doctor').value||'……………………')+'\nNgày lập: '+escape(new Date().toLocaleString('vi-VN'))+'</p>'+
  '<h2>1. Mô tả siêu âm</h2><p>'+escape(result.text||'Nhập dấu hiệu bằng tra cứu HPO.')+'</p><h2>2. Kiểu hình đã duyệt</h2>'+
  table(['Mã HPO','Tiếng Việt','Tiếng Anh','Trạng thái'],result.hpos.map(h=>[h.id,h.vi,h.en,h.status]))+
  '<h2>3. Bệnh được bác sĩ chọn để hội chẩn</h2>'+table(['Hạng','Bệnh / mã','Tương đồng','Kiểu di truyền','Gen'],rows.map(c=>[c.rank,c.disease_name+' · '+c.disease_id,c.match_percentage+'%',c.inheritance_modes.join(', '),c.causative_genes.join(', ')]))+
  '<p>Phần trăm là độ tương đồng kiểu hình, không phải xác suất mắc bệnh.</p><h2>4. Gợi ý xét nghiệm để bác sĩ xem xét</h2>'+
  rows.filter(c=>c.model3_recommended_tests).map(c=>'<p><strong>'+escape(c.disease_name)+'</strong>\n'+escape(c.model3_recommended_tests)+'</p>').join('')+
  '<p>Chỉ định / ghi chú bác sĩ: ........................................................................\n...........................................................................................................</p><div class="signature"><div>Bác sĩ chỉ định<br>(Ký, ghi rõ họ tên)</div><div>Phòng Lab tiếp nhận<br>(Ký, ghi rõ họ tên)</div></div>';
  $('lab').showModal();
};
$('print').onclick=()=>window.print();$('close-lab').onclick=()=>$('lab').close();
$('new-case').onclick=()=>{
  if((chosen.size||$('clinical-text').value)&&!confirm('Bắt đầu ca mới và xóa nội dung ca hiện tại trong phiên này?'))return;
  extractVersion++;searchVersion++;connectionVersion++;clearTimeout(timer);for(const c of requests)c.abort('new-case');selection.clear();extraction=null;searchTerms=[];for(const id of ['case-id','gestation','doctor','clinical-text','query'])$(id).value='';
  renderMentions();renderSearch();invalidate();renderSelected();notice('');localStatus('extract-status','');localStatus('search-status','Nhập tên dấu hiệu để bắt đầu tìm.');
};
$('settings-open').onclick=()=>{$('api-url').value=base;$('settings').showModal();};
$('connect').onclick=e=>{
  e.preventDefault();
  try{
    const value=$('api-url').value.trim().replace(/\/+$/,'');
    if(value){const u=new URL(value);if(u.protocol!=='https:'&&!(u.protocol==='http:'&&['localhost','127.0.0.1'].includes(u.hostname)))throw Error('Máy chủ từ xa phải dùng HTTPS.');if(u.username||u.password||u.search||u.hash)throw Error('Chỉ nhập địa chỉ gốc của máy chủ.');}
    for(const c of requests)c.abort('connection-changed');searchVersion++;extractVersion++;connectionVersion++;clearTimeout(timer);if(value!==base)access='';base=value;if($('access-token').value)access=$('access-token').value;$('access-token').value='';searchTerms=[];extraction=null;renderSearch();renderMentions();
    try{localStorage.setItem('prenatal-api',base);}catch{}
    invalidate();$('settings').close();checkConnection();
  }catch(err){notice(err.message);}
};
async function checkConnection(){
  const seq=connectionVersion;
  try{const status=await api('/api/status',null,15000);if(seq!==connectionVersion)return;$('connection').textContent='● Máy chủ trực tuyến';$('connection').style.color='#286340';if(!status.model_configured)notice('Tra cứu và đối chiếu sẵn sàng. Model trích văn bản chưa được cấu hình.');else if(status.auth_required&&!access)notice('Máy chủ yêu cầu mã truy cập. Mở Kết nối để nhập mã.');else notice('');}
  catch(e){if(seq!==connectionVersion)return;$('connection').textContent='○ Chưa kết nối Ubuntu';notice(e.message);}
}
renderSelected();checkConnection();

