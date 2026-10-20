from neurobench.runner.exec_hard.common import *

CASE_IDS = ('O01', 'O02', 'O03', 'O04')
REVISION = 'open-method-development-20261020-r3'
USER_REQUEST = '不需要考虑暴露问题，模型不可能立刻学习；先优化现有的题目，然后测试'

PROMPT = '''Deliver the research task in context/task.md using the supplied inputs. Read inputs/protocol.json for public data boundaries, acceptance and scoring, and context/CONTRACT.json for the file contract. Choose and justify your methods. No skill-use or tool-count bonus is awarded. Negative results are valid when supported.
Both conditions have identical optional context/skills/ and context/skill_catalog.json, public inputs and generic execution tools. Python calls are fresh processes; files persist. INPUT_DIR and OUTPUT_DIR are pathlib Paths. Only output/ and scratch/ are writable. The delivered entrypoint is output/project/main.py --input INPUT_DIRECTORY --output OUTPUT_DIRECTORY. It must run in a fresh directory using only supplied inputs and delivered project files. context/RUNTIME.txt describes the shared process host.
CPU analysis and model fitting are permitted. No external data, network, credentials, other experiments or evaluator access. Do not modify inputs. These are newly generated synthetic development fixtures, not clinical evidence or an untouched independent validation set. Keep uncertainty and unsupported data visible.
finish_result(status="complete", summary=...) submits actual existing files; the host owns hashes and case identity. inspect_outputs reports structure only. Two submissions maximum, with the second available only after public structural failure; no post-terminal retries or hidden-score feedback. All public acceptance rules are frozen before model dispatch. Additional descriptive narrative is not machine graded for expert scientific acceptance.
Budget per attempt: 900 seconds, 48 model rounds, 192 tools, 120 seconds per Python/project process. Managed LogisticRegression/RidgeClassifier fits are limited to 64; this is not a monitor of every possible custom optimizer. Use finite CPU work within the same wall budget. Report actual validation and avoid unsupported positive claims.
'''
