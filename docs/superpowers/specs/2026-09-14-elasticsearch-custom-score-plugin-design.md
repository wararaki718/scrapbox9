# Elasticsearch Custom Score Plugin Design

## Goal

Create a runnable Elasticsearch plugin sample in `ir_sample/es-plugin-custom-score`. The plugin provides a native score script that multiplies each document's original search score by 100 and can be called from Elasticsearch's standard `script_score` query.

The sample includes a reproducible Gradle build, a Docker image with the plugin installed, Docker Compose lifecycle commands, and documented `curl` requests that demonstrate the score change.

## Scope

The plugin targets Elasticsearch `8.19.21` and Java 21. Elasticsearch plugins require an exact version match, so the Gradle dependencies and Docker image use the same version read from the project's `.env` file.

The script language is `custom_score`. It accepts the fixed source name `multiply_by_100` only in `ScoreScript.CONTEXT`. It has no parameters and does not evaluate arbitrary source text. Supporting configurable multipliers, additional script contexts, and a custom query DSL is outside this sample's scope.

## Architecture

The project is organized by responsibility:

- `src/main/java`: `CustomScorePlugin`, the `ScriptPlugin` entry point, and `CustomScoreScriptEngine`, which compiles the fixed score script.
- `src/test/java`: focused unit tests for score multiplication and script-engine validation.
- `src/main/plugin-metadata/plugin-descriptor.properties`: plugin metadata required by Elasticsearch.
- `build.gradle` and `settings.gradle`: Java compilation, tests, and plugin ZIP packaging.
- `gradlew`, `gradlew.bat`, and `gradle/wrapper`: a pinned Gradle Wrapper.
- `.env`: the single source for the Elasticsearch version used by Gradle and Docker Compose.
- `Dockerfile`: a multi-stage build that creates the plugin ZIP and installs it into the matching Elasticsearch image.
- `compose.yaml`: a single-node Elasticsearch service with security disabled and a health check.
- `Makefile`: concise build, start, stop, and log commands.
- `README.md`: prerequisites, build and lifecycle commands, and manual score-verification requests.

`CustomScorePlugin` registers one `ScriptEngine`. The engine rejects unknown script sources and contexts during compilation. For a valid `multiply_by_100` score script, its leaf factory declares that it needs the original score and returns a script instance whose execution result is `_score * 100`.

## Build and Packaging

The Gradle build uses Java 21 and compiles against the Elasticsearch `8.19.21` artifact without bundling Elasticsearch itself. A `pluginZip` task packages the plugin JAR and descriptor at the ZIP root in the layout expected by `elasticsearch-plugin install`.

`./gradlew test pluginZip` is the direct build command. The equivalent `make build` target is provided for convenience.

The Dockerfile uses a Gradle/JDK 21 builder stage and an Elasticsearch `8.19.21` runtime stage. The runtime stage installs the generated local ZIP with `elasticsearch-plugin install --batch`. Docker Compose passes the version from `.env` into the build and starts the resulting image as a single node. `make up` runs Compose with `--build --wait`, so successful completion means Elasticsearch has reached its healthy state with the plugin installed. `make down` removes the service, and `make logs` follows its logs.

## Demonstration Flow

The README documents these manual steps after `make up`:

1. Confirm the cluster is healthy and the custom plugin is listed.
2. Create a small index with deterministic text documents.
3. Run a normal `match` query and note each hit's `_score`.
4. Wrap the same query in `script_score` using language `custom_score` and source `multiply_by_100`.
5. Compare the returned scores and confirm that each custom score is 100 times its corresponding original score.

The requests use only local HTTP endpoints and require no Elastic Cloud account or credentials.

## Error Handling

The script engine fails compilation with a clear `IllegalArgumentException` when the source is not `multiply_by_100` or the context is not `ScoreScript.CONTEXT`. Build failures stop before an image is produced. An Elasticsearch/plugin version mismatch fails during image construction rather than at query time. The Compose health check prevents `make up` from reporting success before the node can accept requests.

## Testing and Validation

Development follows test-driven development. The initial tests establish that the score operation returns exactly 100 times its input and that the script engine accepts only the supported source and context. Each test is observed failing before its production implementation is added.

Automated validation consists of:

- `./gradlew test` for score and script-engine unit tests.
- `./gradlew pluginZip` for compilation and ZIP assembly.
- Inspection of the ZIP to confirm that the plugin JAR and descriptor are at its root.
- `make up` for a real Elasticsearch startup with the locally built plugin installed and a passing health check.

As requested, HTTP score comparison remains a documented manual `curl` workflow rather than an automated integration assertion. The implementation is complete when the Gradle checks pass, the ZIP has the expected layout, Docker Compose reaches healthy status, and the README commands show the custom query returning scores exactly 100 times the normal query scores.
