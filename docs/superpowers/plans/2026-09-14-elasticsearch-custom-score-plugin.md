# Elasticsearch Custom Score Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Elasticsearch 8.19.21 plugin that exposes a `custom_score` script named `multiply_by_100`, packages it as an installable ZIP, and runs it in Docker Compose.

**Architecture:** A small Java 21 `ScriptPlugin` registers a strict `ScriptEngine` for `ScoreScript.CONTEXT`. A plain Gradle build compiles and tests the plugin and assembles the descriptor and JAR into a ZIP; a multi-stage Dockerfile installs that ZIP into the matching Elasticsearch image.

**Tech Stack:** Java 21, Elasticsearch 8.19.21 plugin API, Gradle 8.10.2, JUnit 5, Docker, Docker Compose, Make

---

### Task 1: Gradle Build Foundation

**Files:**
- Create: `ir_sample/es-plugin-custom-score/settings.gradle`
- Create: `ir_sample/es-plugin-custom-score/build.gradle`
- Create: `ir_sample/es-plugin-custom-score/gradle.properties`
- Create: `ir_sample/es-plugin-custom-score/.env`
- Create: `ir_sample/es-plugin-custom-score/.gitignore`
- Create: `ir_sample/es-plugin-custom-score/gradlew`
- Create: `ir_sample/es-plugin-custom-score/gradlew.bat`
- Create: `ir_sample/es-plugin-custom-score/gradle/wrapper/gradle-wrapper.properties`
- Create: `ir_sample/es-plugin-custom-score/gradle/wrapper/gradle-wrapper.jar`

- [ ] **Step 1: Define the project and version source**

Set `rootProject.name = 'custom-score-plugin'`, set `elasticsearchVersion=8.19.21` in `gradle.properties`, and set `ELASTICSEARCH_VERSION=8.19.21` in `.env`. Ignore `.gradle/` and `build/`.

- [ ] **Step 2: Define Java and test dependencies**

Create a Gradle Java 21 build with Maven Central, `compileOnly "org.elasticsearch:elasticsearch:${elasticsearchVersion}"`, matching `testImplementation`, and `testImplementation 'org.junit.jupiter:junit-jupiter:5.11.4'`. Configure JUnit Platform and build a plugin JAR without bundled Elasticsearch classes.

- [ ] **Step 3: Generate the pinned Gradle Wrapper**

Run: `gradle wrapper --gradle-version 8.10.2`
Expected: wrapper scripts, properties, and `gradle-wrapper.jar` are created.

- [ ] **Step 4: Verify dependency resolution**

Run: `./gradlew dependencies --configuration compileClasspath`
Expected: exit 0 and `org.elasticsearch:elasticsearch:8.19.21` in the dependency tree.

### Task 2: Score Multiplication With TDD

**Files:**
- Create: `ir_sample/es-plugin-custom-score/src/test/java/com/example/elasticsearch/customscore/CustomScoreScriptEngineTest.java`
- Create: `ir_sample/es-plugin-custom-score/src/main/java/com/example/elasticsearch/customscore/CustomScoreScriptEngine.java`

- [ ] **Step 1: Write the failing multiplier test**

Create a JUnit test that asserts `CustomScoreScriptEngine.multiplyBy100(1.25) == 125.0` and `multiplyBy100(0.0) == 0.0`.

- [ ] **Step 2: Run the test and verify RED**

Run: `./gradlew test --tests '*.CustomScoreScriptEngineTest.multipliesScoreBy100'`
Expected: compilation fails because `CustomScoreScriptEngine` does not exist.

- [ ] **Step 3: Add the minimal multiplier implementation**

Create `CustomScoreScriptEngine` implementing `ScriptEngine`, define `LANGUAGE = "custom_score"`, `SOURCE = "multiply_by_100"`, and add package-private `static double multiplyBy100(double score) { return score * 100.0; }`. Stub only required interface methods with unsupported behavior until the next test.

- [ ] **Step 4: Run the multiplier test and verify GREEN**

Run: `./gradlew test --tests '*.CustomScoreScriptEngineTest.multipliesScoreBy100'`
Expected: PASS.

### Task 3: Strict Script Engine With TDD

**Files:**
- Modify: `ir_sample/es-plugin-custom-score/src/test/java/com/example/elasticsearch/customscore/CustomScoreScriptEngineTest.java`
- Modify: `ir_sample/es-plugin-custom-score/src/main/java/com/example/elasticsearch/customscore/CustomScoreScriptEngine.java`

- [ ] **Step 1: Write failing compile-contract tests**

Add tests asserting:

- `getType()` returns `custom_score`.
- `getSupportedContexts()` returns exactly `Set.of(ScoreScript.CONTEXT)`.
- Compiling `multiply_by_100` in `ScoreScript.CONTEXT` returns a `ScoreScript.Factory`.
- An unknown source throws `IllegalArgumentException` containing `Unknown script source`.
- A non-score `ScriptContext` throws `IllegalArgumentException` containing `Unsupported script context`.

- [ ] **Step 2: Run tests and verify RED**

Run: `./gradlew test --tests '*.CustomScoreScriptEngineTest'`
Expected: compile-contract assertions fail because compile and supported contexts are not implemented.

- [ ] **Step 3: Implement the score script factory**

Implement `compile` to validate context and source, then return a `ScoreScript.Factory`. Its leaf factory returns `needs_score() == true`, `needs_termStats() == false`, and creates a `ScoreScript(params, lookup, reader)` whose `execute` calls `multiplyBy100(get_score())`. Return exactly `Set.of(ScoreScript.CONTEXT)` from `getSupportedContexts()`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `./gradlew test --tests '*.CustomScoreScriptEngineTest'`
Expected: all engine tests PASS.

### Task 4: Plugin Entry Point and Descriptor

**Files:**
- Create: `ir_sample/es-plugin-custom-score/src/test/java/com/example/elasticsearch/customscore/CustomScorePluginTest.java`
- Create: `ir_sample/es-plugin-custom-score/src/main/java/com/example/elasticsearch/customscore/CustomScorePlugin.java`
- Create: `ir_sample/es-plugin-custom-score/src/main/plugin-metadata/plugin-descriptor.properties`
- Modify: `ir_sample/es-plugin-custom-score/build.gradle`

- [ ] **Step 1: Write the failing plugin registration test**

Assert that `new CustomScorePlugin().getScriptEngine(Settings.EMPTY, List.of(ScoreScript.CONTEXT))` returns `CustomScoreScriptEngine`.

- [ ] **Step 2: Run the test and verify RED**

Run: `./gradlew test --tests '*.CustomScorePluginTest'`
Expected: compilation fails because `CustomScorePlugin` does not exist.

- [ ] **Step 3: Implement plugin registration**

Create `CustomScorePlugin extends Plugin implements ScriptPlugin` and return a new `CustomScoreScriptEngine` from `getScriptEngine(Settings, Collection<ScriptContext<?>>)`.

- [ ] **Step 4: Add metadata and ZIP packaging**

Create a descriptor with name `custom-score`, plugin version `1.0.0`, Elasticsearch version `8.19.21`, Java version `21`, and classname `com.example.elasticsearch.customscore.CustomScorePlugin`. Add `processPluginMetadata` expansion and a `pluginZip` task that places the processed descriptor and plugin JAR at the ZIP root.

- [ ] **Step 5: Verify tests and ZIP layout**

Run: `./gradlew clean test pluginZip && unzip -l build/distributions/custom-score-1.0.0.zip`
Expected: tests PASS; ZIP lists `custom-score-1.0.0.jar` and `plugin-descriptor.properties` at root.

### Task 5: Docker Runtime and Make Targets

**Files:**
- Create: `ir_sample/es-plugin-custom-score/Dockerfile`
- Create: `ir_sample/es-plugin-custom-score/compose.yaml`
- Create: `ir_sample/es-plugin-custom-score/Makefile`
- Create: `ir_sample/es-plugin-custom-score/.dockerignore`

- [ ] **Step 1: Add the multi-stage image build**

Use `gradle:8.10.2-jdk21` as builder, accept `ELASTICSEARCH_VERSION`, run `gradle --no-daemon clean test pluginZip`, then use `docker.elastic.co/elasticsearch/elasticsearch:${ELASTICSEARCH_VERSION}` and install the ZIP with `bin/elasticsearch-plugin install --batch file:///tmp/custom-score.zip`.

- [ ] **Step 2: Add the single-node Compose service**

Build with `ELASTICSEARCH_VERSION` from `.env`, expose port 9200, set `discovery.type=single-node`, `xpack.security.enabled=false`, JVM heap to 512 MB, and check `/_cluster/health` with the image's available HTTP tool.

- [ ] **Step 3: Add lifecycle commands**

Define `build`, `up`, `down`, and `logs` targets. `up` runs `docker compose up --build --detach --wait`; `down` includes `--volumes --remove-orphans`.

- [ ] **Step 4: Validate the container**

Run: `make up`
Expected: image builds, Compose reports the service healthy, and `curl -fsS http://localhost:9200/_cat/plugins?h=component` includes `custom-score`.

### Task 6: Usage Documentation and End-to-End Check

**Files:**
- Modify: `ir_sample/es-plugin-custom-score/README.md`

- [ ] **Step 1: Document prerequisites and commands**

Document Java 21 for local Gradle use, Docker with Compose, `make build`, `make up`, `make logs`, and `make down`.

- [ ] **Step 2: Document deterministic sample data**

Provide exact `curl` commands to create `custom-score-demo`, index two documents, and call `_refresh`.

- [ ] **Step 3: Document score comparison**

Provide a normal `match` query and the equivalent `script_score` query using `"lang": "custom_score"` and `"source": "multiply_by_100"`. State that corresponding `_score` values in the second response are exactly 100 times those in the first.

- [ ] **Step 4: Run the documented requests**

Run the README commands against the healthy Compose service.
Expected: the plugin appears in `_cat/plugins`, both searches return the same documents in the same order, and each custom `_score` is 100 times its baseline value.

- [ ] **Step 5: Run final verification**

Run: `./gradlew clean test pluginZip && docker compose config --quiet && make down`
Expected: Gradle exits 0, Compose configuration is valid, and containers and volumes are removed.
