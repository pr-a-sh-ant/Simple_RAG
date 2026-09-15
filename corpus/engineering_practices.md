# TechCorp Engineering Practices

This document describes the engineering practices that all TechCorp engineers are expected to follow, covering code review, deployment, on-call, and incident response. These practices support transparency and long-term maintainability.

## Version Control and Branching

Engineers use Git for version control. Branches follow a consistent naming convention:

- `feature/` — New features and enhancements.
- `bugfix/` — Non-urgent bug fixes.
- `hotfix/` — Urgent fixes that must reach production quickly.

All work is merged into the main branch through pull requests. Direct commits to the main branch are not permitted.

## Code Review

Every pull request must be reviewed before it is merged. The code review requirements are:

- At least 1 approval is required for a standard pull request.
- At least 2 approvals are required for production-critical changes, such as changes to authentication, billing, or data handling.
- Pull requests must pass all continuous integration (CI) checks before merging.
- Each pull request must include a clear description of the change and its business purpose.

Reviewers are expected to provide feedback promptly and respectfully, consistent with the Code of Conduct.

## Continuous Integration and Deployment

TechCorp uses GitHub Actions for continuous integration and deployment. Every pull request automatically runs the CI pipeline, which includes linting, unit tests, and integration tests. Changes are first deployed to a staging environment for verification and then promoted to production. Production deployments must be made during business hours whenever possible, except for urgent hotfixes.

## On-Call Rotation

Engineering teams maintain a weekly on-call rotation managed through PagerDuty. The on-call engineer is responsible for responding to alerts and incidents during their rotation. On-call shifts rotate weekly so that responsibility is shared fairly across the team. Engineers are notified before their on-call week begins.

## Incident Severity Levels

Incidents are classified by severity so that the team can respond appropriately:

- **SEV1** — Critical incident such as a full outage or data loss. Requires acknowledgement within 15 minutes and an immediate response.
- **SEV2** — Major incident such as a significant feature being unavailable. Requires a prompt response during business hours.
- **SEV3** — Minor incident with limited impact and an available workaround.

## Postmortems

A blameless postmortem is required for every SEV1 and SEV2 incident. The postmortem documents the timeline, root cause, and follow-up action items. Postmortems focus on improving systems and processes rather than assigning blame to individuals. Completed postmortems are shared with the engineering organization to support continuous learning.

This document is a fictional sample created for educational and demonstration purposes.
