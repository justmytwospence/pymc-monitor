# pymc-monitor development

This repository contains local sampler logic only. Keep Modal transport and
cloud/cache dependencies out of `src/pymc_monitor`. The ignored
`.extraction-source/` snapshot is a read-only reference and must never enter
an artifact. Test both supported PyMC/ArviZ major pairs. Keep live posterior
diagnostics in constrained variable space and attach final reports based on
recovered inference data. Never silently disable adaptive stopping after a
sampling or diagnostic error. Use `scripts/lock.sh` to update `uv.lock`.
