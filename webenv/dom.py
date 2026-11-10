"""Declarative phase and terminal projection for visible DOM observations."""
import re


def terminal(headings, config):
    rules=config.get('terminal') or {}
    for success,key in [(True,'success_headings'),(False,'failure_headings')]:
        for pattern in rules.get(key,[]):
            if any(re.fullmatch(pattern,str(text).strip()) for text in headings):
                return True,success
    return False,False


def phase(frame, config):
    for rule in config.get('phases',[]):
        source='\n'.join(frame.get('headings',[])) if rule.get('field')=='headings' else frame.get('text','')
        if re.search(rule['pattern'],source):return rule['name']
    return ''


def annotate_candidates(candidates, flags, rules):
    """Attach declared public control semantics without choosing an action."""
    result=[]
    for original in candidates:
        item=dict(original)
        for rule in rules:
            if rule.get('flag') and not flags.get(rule['flag']):continue
            if re.search(rule['pattern'],item.get('label','')):
                item['label']=rule.get('prefix','')+item.get('label','')
                item['role']=rule.get('role','')
                break
        result.append(item)
    return result
