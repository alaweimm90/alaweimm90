.PHONY: qap-bench orchestrator-up orchestrator-test

qap-bench:
	@if [ -z "$$QAPLIB_DATA_DIR" ]; then echo "Set QAPLIB_DATA_DIR to directory with .dat files"; exit 1; fi
	@QAPFLOW_BACKEND=external \
		python -m libria_meta.cli.qapflow_cli --bench qaplib --modes hybrid,nesterov --time-limit 30 --out bench_results.json

orchestrator-up:
	@python -m libria_meta.services.mezan_orchestrator

orchestrator-test:
	@python - << 'PY'
import json, urllib.request
payload = {"fit": [[1,2],[2,1]]}
req = urllib.request.Request('http://127.0.0.1:8081/qapflow/solve', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json'})
resp = urllib.request.urlopen(req, timeout=5)
	print(resp.read().decode('utf-8'))
PY

.PHONY: issues-csv
issues-csv:
	@python scripts/generate_github_issues.py > issues.csv && echo "Wrote issues.csv"

.PHONY: docker-build docker-run
docker-build:
	@docker build -t mezan-orchestrator -f Dockerfile .

docker-run:
	@docker run --rm -p 8081:8081 mezan-orchestrator

.PHONY: bench-html
bench-html:
	@if [ -z "$$QAPLIB_DATA_DIR" ]; then echo "Set QAPLIB_DATA_DIR"; exit 1; fi
	@QAPFLOW_BACKEND=external \
		python -m libria_meta.cli.qapflow_cli --bench qaplib --modes hybrid,nesterov --time-limit 30 --out results.json --html report.html && echo "Wrote report.html"

.PHONY: optibench-replay
optibench-replay:
	@if [ -z "$$QAPLIB_DATA_DIR" ]; then echo "Set QAPLIB_DATA_DIR"; exit 1; fi
	@python -m optibench.replay --entry qaplib-mini --out results/optibench/qaplib-mini.json
