"""Opt-in executable development cases, independent of frozen model campaigns.

Build: python -m neurobench.runner.exec_dev.build --destination <new packet directory>
Check: python -m neurobench.runner.exec_dev.qualify --packet <packet> --destination <new run>

Only public/cases/<id> belongs in a future model workspace. Reference solutions,
qualification mutations, trusted tool receipts and oracle code are evaluator
material. This is development evidence, not a blind scientific acceptance set.
No module imports a model client or launches a model/judge request.
"""
