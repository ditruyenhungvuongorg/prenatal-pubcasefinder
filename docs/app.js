'use strict';
const $=id=>document.getElementById(id);
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let base=window.PRENATAL_CONFIG?.apiBase||'';
try {base=localStorage.getItem('prenatal-api')||base;} catch {}
let access='', chosen=new Map(), result=null, revision=0, searchVersion=0, extractVersion=0, busy=false;
const labIds=new Set();
function notice(message=''){ $('notice').textContent=message;$('notice').hidden=!message; }
async function api(path,payload,timeout=120000){
  if(!base && location.hostname.endsWith('github.io')) throw Error('Chưa cấu hình máy chủ Ubuntu. Mở Kết nối để nhập địa chỉ API.');
  const response=await fetch(base+path,{method:payload?'POST':'GET',headers:{...(payload?{'Content-Type':'application/json'}:{}),...(access?{Authorization:'Bearer '+access}:{})},...(payload?{body:JSON.stringify(payload)}:{}),signal:AbortSignal.timeout(timeout),cache:'no-store'});
  let data;try{data=await response.json();}catch{throw Error('Máy chủ không trả về dữ liệu hợp lệ. Kiểm tra địa chỉ API.');}
  if(!response.ok)throw Error(data.error||'Máy chủ không thể xử lý yêu cầu.');
  return data;
}
function updateActions(){$('match').disabled=busy||!chosen.size||!$('reviewed').checked;$('export').disabled=!result||!labIds.size;}
function invalidate(){
  revision++;result=null;labIds.clear();$('reviewed').checked=false;$('pattern').hidden=true;
  $('results-title').textContent='Kết quả bệnh tiềm năng';
  $('results').innerHTML='<div class="empty"><h3>Danh sách đã thay đổi</h3><p>Duyệt HPO và đối chiếu lại để tạo kết quả mới.</p></div>';
  updateActions();
}
function add(term,state='CÓ'){
  if(chosen.has(term.id)){notice('Mã '+term.id+' đã có trong danh sách.');return;}
  chosen.set(term.id,{...term,status:state});invalidate();renderSelected();
}
function renderSelected(){
  $('hpo-count').textContent=chosen.size;
  $('selected').innerHTML=chosen.size?'':'<p class="hint">Chưa có HPO. Hai cách nhập sẽ cùng bổ sung vào đây.</p>';
  for(const h of chosen.values()){
    const div=document.createElement('div');div.className='chosen';
    div.innerHTML='<div class="chosen-head"><div><strong>'+escape(h.vi||h.en)+'</strong><small>'+escape(h.en)+' · '+escape(h.id)+'</small></div><button class="text-button" aria-label="Xóa '+escape(h.id)+'">Xóa</button></div><select aria-label="Trạng thái '+escape(h.id)+'">'+['CÓ','NGHI NGỜ','KHÔNG'].map(s=>'<option '+(s===h.status?'selected':'')+'>'+s+'</option>').join('')+'</select>';
    div.querySelector('button').onclick=()=>{chosen.delete(h.id);invalidate();renderSelected();};
    div.querySelector('select').onchange=e=>{h.status=e.target.value;invalidate();};
    $('selected').append(div);
  }updateActions();
}
for(const mode of ['search','text']) $('tab-'+mode).onclick=()=>{
  $('manual').hidden=mode!=='search';$('paragraph').hidden=mode!=='text';
  for(const m of ['search','text'])$('tab-'+m).setAttribute('aria-selected',String(m===mode));
  $(mode==='search'?'query':'clinical-text').focus();
};
let timer;
$('query').oninput=()=>{
  clearTimeout(timer);const seq=++searchVersion,q=$('query').value.trim();$('search-results').replaceChildren();
  if(!q)return;
  timer=setTimeout(async()=>{
    try{
      const terms=await api('/api/hpo_search?q='+encodeURIComponent(q),null,20000);
      if(seq!==searchVersion)return;
      $('search-results').innerHTML=terms.length?'':'<p class="hint">Không tìm thấy. Thử cụm ngắn hơn hoặc mã HPO.</p>';
      for(const term of terms){
        const b=document.createElement('button');b.className='search-item';
        b.innerHTML=escape(term.vi||term.en)+'<small>'+escape(term.en)+' · '+escape(term.id)+'</small>';
        b.onclick=()=>{add(term);$('query').focus();};$('search-results').append(b);
      }
    }catch(e){if(seq===searchVersion)notice(e.message);}
  },220);
};
$('clinical-text').oninput=()=>{extractVersion++;$('mentions').replaceChildren();invalidate();};
$('extract').onclick=async()=>{
  const text=$('clinical-text').value.trim();if(!text){notice('Nhập đoạn mô tả siêu âm trước.');return;}
  const seq=++extractVersion;$('extract').disabled=true;notice('Đang nhận diện dấu hiệu bằng Model 1 v3.8…');
  try{
    const data=await api('/api/extract_hpo',{text});
    if(seq!==extractVersion)return;
    $('mentions').replaceChildren();
    notice(data.mentions.length?'Chọn HPO phù hợp cho từng cụm, sau đó kiểm tra trạng thái trong danh sách.':'Model không trích được dấu hiệu. Bạn có thể tìm HPO thủ công.');
    for(const m of data.mentions){
      const box=document.createElement('div');box.className='mention';
      box.innerHTML='<blockquote>“'+escape(m.mention_text)+'”</blockquote><p class="hint">'+escape(m.explanation)+(m.context_review?' · Có phủ định, tiền sử hoặc chủ thể cần kiểm tra.':'')+'</p>';
      if(!m.candidates.length)box.innerHTML+='<p class="hint">Chưa tìm được HPO tương ứng; tìm thủ công bằng cụm ngắn hơn.</p>';
      for(const term of m.candidates){
        const b=document.createElement('button');b.className='search-item';b.textContent=(term.vi||term.en)+' · '+term.id+' +';
        b.onclick=()=>add(term,m.context_review?'NGHI NGỜ':m.status);box.append(b);
      }
      $('mentions').append(box);
    }
  }catch(e){notice(e.message);}finally{$('extract').disabled=false;}
};
$('clear-hpos').onclick=()=>{chosen.clear();invalidate();renderSelected();};
$('reviewed').onchange=updateActions;
$('match').onclick=async()=>{
  if(!$('reviewed').checked)return;
  const seq=revision;busy=true;updateActions();notice('Đang đối chiếu hồ sơ bệnh hiếm…');
  try{
    const data=await api('/api/match_diseases',{hpos:[...chosen.values()].map(h=>({id:h.id,status:h.status}))});
    if(seq!==revision){notice('Ca đã thay đổi trong khi xử lý. Vui lòng đối chiếu lại.');return;}
    result={...data,hpos:structuredClone([...chosen.values()]),text:$('clinical-text').value};
    labIds.clear();for(const c of data.candidates.slice(0,5))labIds.add(c.disease_id);
    const modes=[...new Set(data.candidates.flatMap(c=>c.inheritance_modes))].sort();
    $('inheritance-filter').innerHTML='<option value="">Tất cả kiểu di truyền</option>'+modes.map(m=>'<option>'+escape(m)+'</option>').join('');
    renderResults();notice('');
  }catch(e){result=null;labIds.clear();notice(e.message);}
  finally{busy=false;updateActions();}
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
      b.onclick=()=>{add(t,'NGHI NGỜ');notice('Đã thêm dấu hiệu ở trạng thái Nghi ngờ. Bác sĩ xác nhận sau khi kiểm tra.');};
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
  extractVersion++;searchVersion++;chosen.clear();for(const id of ['case-id','gestation','doctor','clinical-text','query'])$(id).value='';
  $('mentions').replaceChildren();$('search-results').replaceChildren();invalidate();renderSelected();notice('');
};
$('settings-open').onclick=()=>{$('api-url').value=base;$('settings').showModal();};
$('connect').onclick=e=>{
  e.preventDefault();
  try{
    const value=$('api-url').value.trim().replace(/\/+$/,'');
    if(value){const u=new URL(value);if(u.protocol!=='https:'&&!['localhost','127.0.0.1'].includes(u.hostname))throw Error('Máy chủ từ xa phải dùng HTTPS.');if(u.username||u.password||u.search||u.hash)throw Error('Chỉ nhập địa chỉ gốc của máy chủ.');}
    base=value;access=$('access-token').value;$('access-token').value='';
    try{localStorage.setItem('prenatal-api',base);}catch{}
    invalidate();$('settings').close();checkConnection();
  }catch(err){notice(err.message);}
};
async function checkConnection(){
  try{const status=await api('/api/status',null,15000);$('connection').textContent='● Máy chủ trực tuyến';$('connection').style.color='#286340';if(!status.model_configured)notice('Tra cứu và đối chiếu sẵn sàng. Model trích văn bản chưa được cấu hình.');else if(status.auth_required&&!access)notice('Máy chủ yêu cầu mã truy cập. Mở Kết nối để nhập mã.');else notice('');}
  catch(e){$('connection').textContent='○ Chưa kết nối Ubuntu';notice(e.message);}
}
renderSelected();checkConnection();

