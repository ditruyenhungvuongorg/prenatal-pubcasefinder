(function (root) {
  'use strict';
  const states = new Set(['CÓ', 'NGHI NGỜ', 'KHÔNG']);
  const sourceKey = source => JSON.stringify([source.kind, source.text || '', source.phrase || '', source.start ?? null]);
  class CaseSelection {
    constructor() { this.items = new Map(); }
    add(term, status = 'CÓ', source = {kind: 'manual'}) {
      if (!/^HP:\d{7}$/.test(term.id) || !states.has(status)) throw new Error('Mã HPO hoặc trạng thái không hợp lệ.');
      const old = this.items.get(term.id);
      const evidence = {...source, key: sourceKey(source)};
      if (!old) {
        this.items.set(term.id, {...term, status, sources: [evidence], pending: [], resolved: []});
        return 'added';
      }
      if (!old.sources.some(s => s.key === evidence.key)) old.sources.push(evidence);
      // A manual search selects an existing term; it does not assert a new status.
      if (source.kind === 'manual' || status === old.status) return 'duplicate';
      const decision = JSON.stringify([evidence.key, status, old.status]);
      if (old.resolved.includes(decision)) return 'duplicate';
      if (!old.pending.some(p => p.decision === decision)) old.pending.push({status, decision, phrase: source.phrase || ''});
      return 'conflict';
    }
    resolve(id, status) {
      if (!states.has(status)) throw new Error('Trạng thái không hợp lệ.');
      const item = this.items.get(id);
      if (!item) return;
      item.resolved.push(...item.pending.map(p => p.decision));
      item.pending = [];
      item.status = status;
    }
    get conflicts() { return [...this.items.values()].filter(h => h.pending.length).length; }
    stale(id, text) { return this.items.get(id)?.sources.some(s => s.kind === 'text' && s.text !== text) || false; }
    remove(id) { this.items.delete(id); }
    clear() { this.items.clear(); }
  }
  root.CaseSelection = CaseSelection;
  if (typeof module !== 'undefined' && module.exports) module.exports = {CaseSelection};
})(globalThis);
