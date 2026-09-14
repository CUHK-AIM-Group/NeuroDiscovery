"""Freeze the original runtime boundary before authorized R41 compatibility edits."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require

OUTPUT=j.OUTPUT/'round41_bulk_cleanup'

if __name__=='__main__':
    b=j.read_json(OUTPUT/'SOURCE_BOUNDARY.json');c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c==b['campaign'] and c['active_process'] is None,'source advanced')
    inspection=j.read_json(OUTPUT/'SOURCE_INSPECTION.json')
    require(inspection['graph']==c['current_graph'] and inspection['full_source_sha_verified'],'missing current full scan')
    for fp in b['code']:require(j.fingerprint(fp['path'])==fp,'runtime changed before preparation')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    c.update(status='MANUAL_PREPARING',phase='R41批量metadata精简与相关关系索引：兼容性测试中',updated_at=j.utc_now())
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    print('R41_PREPARING_NO_KG_WRITER',flush=True)
