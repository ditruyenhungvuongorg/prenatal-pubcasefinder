const {test}=require('node:test');
const assert=require('node:assert/strict');
const {CaseSelection}=require('./docs/case-state.js');
const term={id:'HP:0000252',vi:'Tật đầu nhỏ',en:'Microcephaly'};
const text={kind:'text',text:'nghi đầu nhỏ',phrase:'đầu nhỏ',start:4};
test('manual and extraction share one record, rerun preserves selection',()=>{
 const s=new CaseSelection();s.add(term);assert.equal(s.add(term,'CÓ',text),'duplicate');
 s.add(term,'CÓ',text);assert.equal(s.items.size,1);assert.equal(s.items.get(term.id).sources.length,2);
});
test('different status creates conflict without overwriting doctor selection',()=>{
 const s=new CaseSelection();s.add(term);assert.equal(s.add(term,'NGHI NGỜ',text),'conflict');
 assert.equal(s.items.get(term.id).status,'CÓ');assert.equal(s.conflicts,1);
 s.add(term,'NGHI NGỜ',text);assert.equal(s.items.get(term.id).pending.length,1);
});
test('doctor may keep current status; same extraction does not reopen conflict',()=>{
 const s=new CaseSelection();s.add(term);s.add(term,'NGHI NGỜ',text);s.resolve(term.id,'CÓ');
 assert.equal(s.conflicts,0);assert.equal(s.add(term,'NGHI NGỜ',text),'duplicate');
});
test('doctor may adopt proposed status and manual search never resets it',()=>{
 const s=new CaseSelection();s.add(term);s.add(term,'NGHI NGỜ',text);s.resolve(term.id,'NGHI NGỜ');
 s.add(term);assert.equal(s.items.get(term.id).status,'NGHI NGỜ');assert.equal(s.conflicts,0);
});
test('edited paragraph flags old evidence without deleting selected HPO',()=>{
 const s=new CaseSelection();s.add(term,'CÓ',text);assert.equal(s.stale(term.id,text.text),false);
 assert.equal(s.stale(term.id,'văn bản mới'),true);assert.equal(s.items.size,1);
});
test('remove and new case clear conflicts and evidence',()=>{
 const s=new CaseSelection();s.add(term);s.add(term,'NGHI NGỜ',text);s.remove(term.id);
 assert.equal(s.conflicts,0);s.add(term);s.clear();assert.equal(s.items.size,0);
});
test('invalid status and IDs are rejected',()=>{
 const s=new CaseSelection();assert.throws(()=>s.add(term,'UNKNOWN'));assert.throws(()=>s.add({id:'x'}));
});

