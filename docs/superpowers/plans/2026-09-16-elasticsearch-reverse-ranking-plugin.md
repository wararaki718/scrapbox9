# Elasticsearch Reverse Ranking Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Elasticsearch 8.19.21 plugin that takes the original top N hits on a single shard, replaces positive scores with their reciprocals, keeps zero at zero, and verifies the resulting order automatically.

**Architecture:** Register a custom `reverse_rank` rescorer through `SearchPlugin.RescorerSpec`. A stateless builder parses the empty rescorer body and creates a `RescoreContext`; a stateless rescorer mutates and sorts Lucene `TopDocs`. Gradle packages the plugin, Docker installs it, and a shell script drives a deterministic one-shard integration check.

**Tech Stack:** Java 21, Elasticsearch 8.19.21 plugin API, Lucene `TopDocs`, Gradle 8.10.2, JUnit 5, Docker Compose, Bash, Python 3

---

## File Structure

- `ir_sample/es-plugin-reverse-ranking/build.gradle`: Java compilation, dependencies, and plugin ZIP.
- `ir_sample/es-plugin-reverse-ranking/src/main/plugin-metadata/plugin-descriptor.properties`: plugin descriptor.
- `ir_sample/es-plugin-reverse-ranking/src/main/java/com/example/elasticsearch/reverseranking/ReverseRankingPlugin.java`: rescorer registration.
- `ir_sample/es-plugin-reverse-ranking/src/main/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorerBuilder.java`: parsing, serialization, and context construction.
- `ir_sample/es-plugin-reverse-ranking/src/main/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorer.java`: reciprocal scoring, sorting, and explanations.
- `ir_sample/es-plugin-reverse-ranking/src/test/java/com/example/elasticsearch/reverseranking/*Test.java`: focused unit tests.
- `ir_sample/es-plugin-reverse-ranking/Dockerfile` and `compose.yaml`: plugin-enabled Elasticsearch.
- `ir_sample/es-plugin-reverse-ranking/scripts/verify.sh`: HTTP-level behavior check.
- `ir_sample/es-plugin-reverse-ranking/Makefile`: build and runtime commands.
- `ir_sample/es-plugin-reverse-ranking/README.md`: usage and constraints.

### Task 1: Gradle Plugin Skeleton

**Files:**
- Create: `ir_sample/es-plugin-reverse-ranking/.env`
- Create: `ir_sample/es-plugin-reverse-ranking/.gitignore`
- Create: `ir_sample/es-plugin-reverse-ranking/build.gradle`
- Create: `ir_sample/es-plugin-reverse-ranking/settings.gradle`
- Create: `ir_sample/es-plugin-reverse-ranking/gradlew`
- Create: `ir_sample/es-plugin-reverse-ranking/gradlew.bat`
- Create: `ir_sample/es-plugin-reverse-ranking/gradle/wrapper/gradle-wrapper.jar`
- Create: `ir_sample/es-plugin-reverse-ranking/gradle/wrapper/gradle-wrapper.properties`
- Create: `ir_sample/es-plugin-reverse-ranking/src/main/plugin-metadata/plugin-descriptor.properties`

- [ ] **Step 1: Copy the tested Gradle wrapper from the neighboring sample**

```bash
cd ir_sample/es-plugin-reverse-ranking
cp ../es-plugin-custom-score/gradlew ../es-plugin-custom-score/gradlew.bat .
mkdir -p gradle/wrapper
cp ../es-plugin-custom-score/gradle/wrapper/gradle-wrapper.jar gradle/wrapper/
cp ../es-plugin-custom-score/gradle/wrapper/gradle-wrapper.properties gradle/wrapper/
chmod +x gradlew
```

Expected: the Gradle 8.10.2 wrapper files exist locally.

- [ ] **Step 2: Add project configuration**

Create `.env`:

```dotenv
ELASTICSEARCH_VERSION=8.19.21
```

Create `.gitignore`:

```gitignore
.gradle/
build/
```

Create `settings.gradle`:

```groovy
rootProject.name = 'reverse-ranking-plugin'
```

Create `build.gradle`:

```groovy
plugins {
    id 'java'
}

group = 'com.example.elasticsearch'
version = '1.0.0'

def environment = file('.env').readLines()
    .findAll { it && !it.startsWith('#') }
    .collectEntries { line ->
        def parts = line.split('=', 2)
        [(parts[0]): parts[1]]
    }
def elasticsearchVersion = environment.ELASTICSEARCH_VERSION

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
}

dependencies {
    compileOnly "org.elasticsearch:elasticsearch:${elasticsearchVersion}"
    testImplementation "org.elasticsearch:elasticsearch:${elasticsearchVersion}"
    testImplementation 'org.junit.jupiter:junit-jupiter:5.11.4'
}

tasks.withType(Test).configureEach {
    useJUnitPlatform()
}

def processPluginMetadata = tasks.register('processPluginMetadata', Copy) {
    from('src/main/plugin-metadata') {
        expand(pluginVersion: project.version, elasticsearchVersion: elasticsearchVersion)
    }
    into(layout.buildDirectory.dir('generated-resources/plugin-metadata'))
}

tasks.register('pluginZip', Zip) {
    dependsOn(tasks.named('jar'), processPluginMetadata)
    archiveBaseName = 'reverse-ranking'
    archiveVersion = project.version
    destinationDirectory = layout.buildDirectory.dir('distributions')
    from(tasks.named('jar'))
    from(processPluginMetadata)
}
```

Create `src/main/plugin-metadata/plugin-descriptor.properties`:

```properties
description=Re-ranks top Elasticsearch hits using reciprocal scores
version=${pluginVersion}
name=reverse-ranking
classname=com.example.elasticsearch.reverseranking.ReverseRankingPlugin
java.version=21
elasticsearch.version=${elasticsearchVersion}
```

- [ ] **Step 3: Verify the empty project configuration**

```bash
./gradlew test pluginZip
```

Expected: `BUILD SUCCESSFUL` and `build/distributions/reverse-ranking-1.0.0.zip` exists.

- [ ] **Step 4: Commit the build skeleton**

```bash
git add ir_sample/es-plugin-reverse-ranking/.env ir_sample/es-plugin-reverse-ranking/.gitignore
git add ir_sample/es-plugin-reverse-ranking/build.gradle ir_sample/es-plugin-reverse-ranking/settings.gradle
git add ir_sample/es-plugin-reverse-ranking/gradlew ir_sample/es-plugin-reverse-ranking/gradlew.bat ir_sample/es-plugin-reverse-ranking/gradle
git add ir_sample/es-plugin-reverse-ranking/src/main/plugin-metadata/plugin-descriptor.properties
git commit -m "build: scaffold reverse ranking plugin"
```

### Task 2: Reciprocal Score Rescorer

**Files:**
- Create: `ir_sample/es-plugin-reverse-ranking/src/test/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorerTest.java`
- Create: `ir_sample/es-plugin-reverse-ranking/src/main/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorer.java`

- [ ] **Step 1: Write the failing tests**

Create `ReverseRankingRescorerTest.java`:

```java
package com.example.elasticsearch.reverseranking;

import org.apache.lucene.search.ScoreDoc;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ReverseRankingRescorerTest {
    @Test
    void convertsPositiveScoresToReciprocalsAndKeepsZeroAtZero() {
        assertEquals(0.25f, ReverseRankingRescorer.reciprocal(4.0f));
        assertEquals(0.0f, ReverseRankingRescorer.reciprocal(0.0f));
    }

    @Test
    void rejectsNegativeAndNonFiniteScores() {
        assertThrows(IllegalArgumentException.class, () -> ReverseRankingRescorer.reciprocal(-1.0f));
        assertThrows(IllegalArgumentException.class, () -> ReverseRankingRescorer.reciprocal(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> ReverseRankingRescorer.reciprocal(Float.POSITIVE_INFINITY));
    }

    @Test
    void transformsTheWindowAndSortsByNewScore() {
        var scoreDocs = new ScoreDoc[] {
            new ScoreDoc(40, 4.0f),
            new ScoreDoc(30, 3.0f),
            new ScoreDoc(20, 2.0f),
            new ScoreDoc(10, 1.0f)
        };

        ReverseRankingRescorer.rescore(scoreDocs, 3);

        assertArrayEquals(new int[] { 10, 20, 30, 40 }, java.util.Arrays.stream(scoreDocs).mapToInt(hit -> hit.doc).toArray());
        assertArrayEquals(new float[] { 1.0f, 0.5f, 1.0f / 3.0f, 0.25f }, scores(scoreDocs));
    }

    private static float[] scores(ScoreDoc[] scoreDocs) {
        var scores = new float[scoreDocs.length];
        for (int index = 0; index < scoreDocs.length; index++) {
            scores[index] = scoreDocs[index].score;
        }
        return scores;
    }
}
```

Add an explanation test using `new RescoreContext(3, ReverseRankingRescorer.INSTANCE)`, mark the document as rescored with `context.setRescoredDocs(Set.of(20))`, pass `Explanation.match(2.0f, "original")`, and assert that the returned explanation is a match with value `0.5f`, description `reciprocal of the original score`, and the original explanation as its detail. Also assert that an unrescored document returns the original explanation unchanged.

- [ ] **Step 2: Run the test to verify RED**

```bash
./gradlew test --tests '*ReverseRankingRescorerTest'
```

Expected: compilation fails because `ReverseRankingRescorer` does not exist.

- [ ] **Step 3: Implement the stateless rescorer**

Create `ReverseRankingRescorer.java`:

```java
package com.example.elasticsearch.reverseranking;

import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.elasticsearch.search.rescore.RescoreContext;
import org.elasticsearch.search.rescore.Rescorer;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

final class ReverseRankingRescorer implements Rescorer {
    static final ReverseRankingRescorer INSTANCE = new ReverseRankingRescorer();

    private ReverseRankingRescorer() {}

    static float reciprocal(float score) {
        if (Float.isFinite(score) == false || score < 0.0f) {
            throw new IllegalArgumentException("score must be finite and non-negative: " + score);
        }
        return score == 0.0f ? 0.0f : 1.0f / score;
    }

    static void rescore(ScoreDoc[] scoreDocs, int windowSize) {
        int rescoreCount = Math.min(windowSize, scoreDocs.length);
        for (int index = 0; index < rescoreCount; index++) {
            scoreDocs[index].score = reciprocal(scoreDocs[index].score);
        }
        Arrays.sort(scoreDocs, Rescorer.SCORE_DOC_COMPARATOR);
    }

    @Override
    public TopDocs rescore(TopDocs topDocs, IndexSearcher searcher, RescoreContext context) throws IOException {
        if (topDocs == null || topDocs.scoreDocs.length == 0) {
            return topDocs;
        }
        int count = Math.min(context.getWindowSize(), topDocs.scoreDocs.length);
        context.setRescoredDocs(
            IntStream.range(0, count).map(index -> topDocs.scoreDocs[index].doc).boxed().collect(java.util.stream.Collectors.toSet())
        );
        rescore(topDocs.scoreDocs, context.getWindowSize());
        return topDocs;
    }

    @Override
    public Explanation explain(int docId, IndexSearcher searcher, RescoreContext context, Explanation source) throws IOException {
        if (source == null || source.isMatch() == false || context.isRescored(docId) == false) {
            return source;
        }
        return Explanation.match(reciprocal(source.getValue().floatValue()), "reciprocal of the original score", source);
    }
}
```

- [ ] **Step 4: Run the focused test to verify GREEN**

```bash
./gradlew test --tests '*ReverseRankingRescorerTest'
```

Expected: all tests pass.

- [ ] **Step 5: Commit the core behavior**

```bash
git add ir_sample/es-plugin-reverse-ranking/src/main/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorer.java
git add ir_sample/es-plugin-reverse-ranking/src/test/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorerTest.java
git commit -m "feat: add reciprocal score rescorer"
```

### Task 3: Builder And Plugin Registration

**Files:**
- Create: `ir_sample/es-plugin-reverse-ranking/src/test/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorerBuilderTest.java`
- Create: `ir_sample/es-plugin-reverse-ranking/src/test/java/com/example/elasticsearch/reverseranking/ReverseRankingPluginTest.java`
- Create: `ir_sample/es-plugin-reverse-ranking/src/main/java/com/example/elasticsearch/reverseranking/ReverseRankingRescorerBuilder.java`
- Create: `ir_sample/es-plugin-reverse-ranking/src/main/java/com/example/elasticsearch/reverseranking/ReverseRankingPlugin.java`

- [ ] **Step 1: Write failing builder and registration tests**

Tests must cover these exact calls:

```java
@Test
void parsesAnEmptyConfiguration() throws IOException {
    try (XContentParser parser = JsonXContent.jsonXContent.createParser(
        NamedXContentRegistry.EMPTY, LoggingDeprecationHandler.INSTANCE, "{}"
    )) {
        parser.nextToken();
        assertInstanceOf(ReverseRankingRescorerBuilder.class, ReverseRankingRescorerBuilder.fromXContent(parser));
    }
}

@Test
void rejectsUnknownConfigurationFields() throws IOException {
    try (XContentParser parser = JsonXContent.jsonXContent.createParser(
        NamedXContentRegistry.EMPTY, LoggingDeprecationHandler.INSTANCE, "{\"unknown\":true}"
    )) {
        parser.nextToken();
        assertThrows(ParsingException.class, () -> ReverseRankingRescorerBuilder.fromXContent(parser));
    }
}

@Test
void requiresAPositiveWindowSize() {
    assertThrows(IllegalArgumentException.class, () -> new ReverseRankingRescorerBuilder().windowSize(0));
}

@Test
void registersReverseRankRescorer() {
    var specs = new ReverseRankingPlugin().getRescorers();
    assertEquals(1, specs.size());
    assertEquals("reverse_rank", specs.getFirst().getName().getPreferredName());
}
```

Also serialize a builder with `windowSize(3)` through `BytesStreamOutput` and construct a copy with `ReverseRankingRescorerBuilder(StreamInput)`; assert the copied window is 3.

- [ ] **Step 2: Run tests to verify RED**

```bash
./gradlew test --tests '*ReverseRankingRescorerBuilderTest' --tests '*ReverseRankingPluginTest'
```

Expected: compilation fails because the builder and plugin classes do not exist.

- [ ] **Step 3: Implement the builder**

Create `ReverseRankingRescorerBuilder.java`:

```java
package com.example.elasticsearch.reverseranking;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.search.rescore.RescoreContext;
import org.elasticsearch.search.rescore.RescorerBuilder;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;

public final class ReverseRankingRescorerBuilder extends RescorerBuilder<ReverseRankingRescorerBuilder> {
    public static final String NAME = "reverse_rank";

    public ReverseRankingRescorerBuilder() {}

    public ReverseRankingRescorerBuilder(StreamInput input) throws IOException {
        super(input);
    }

    public static ReverseRankingRescorerBuilder fromXContent(XContentParser parser) throws IOException {
        XContentParser.Token token;
        String fieldName = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                fieldName = parser.currentName();
            } else {
                throw new ParsingException(parser.getTokenLocation(), NAME + " does not support [" + fieldName + "]");
            }
        }
        return new ReverseRankingRescorerBuilder();
    }

    @Override
    public ReverseRankingRescorerBuilder windowSize(int windowSize) {
        if (windowSize <= 0) {
            throw new IllegalArgumentException("window_size must be greater than 0");
        }
        return super.windowSize(windowSize);
    }

    @Override
    protected boolean isWindowSizeRequired() {
        return true;
    }

    @Override
    protected void doWriteTo(StreamOutput output) throws IOException {}

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME).endObject();
    }

    @Override
    protected RescoreContext innerBuildContext(int windowSize, SearchExecutionContext context) throws IOException {
        return new RescoreContext(windowSize, ReverseRankingRescorer.INSTANCE);
    }

    @Override
    public RescorerBuilder<ReverseRankingRescorerBuilder> rewrite(QueryRewriteContext context) throws IOException {
        return this;
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersion.minimumCompatible();
    }
}
```

- [ ] **Step 4: Implement plugin registration**

Create `ReverseRankingPlugin.java`:

```java
package com.example.elasticsearch.reverseranking;

import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;

import java.util.List;

public final class ReverseRankingPlugin extends Plugin implements SearchPlugin {
    @Override
    public List<RescorerSpec<?>> getRescorers() {
        return List.of(new RescorerSpec<>(
            ReverseRankingRescorerBuilder.NAME,
            ReverseRankingRescorerBuilder::new,
            ReverseRankingRescorerBuilder::fromXContent
        ));
    }
}
```

- [ ] **Step 5: Run focused and full tests**

```bash
./gradlew test --tests '*ReverseRankingRescorerBuilderTest' --tests '*ReverseRankingPluginTest'
./gradlew test
```

Expected: focused tests and full suite pass.

- [ ] **Step 6: Commit registration support**

```bash
git add ir_sample/es-plugin-reverse-ranking/src/main ir_sample/es-plugin-reverse-ranking/src/test
git commit -m "feat: register reverse rank rescorer"
```

### Task 4: Docker Runtime And Behavior Verification

**Files:**
- Create: `ir_sample/es-plugin-reverse-ranking/.dockerignore`
- Create: `ir_sample/es-plugin-reverse-ranking/Dockerfile`
- Create: `ir_sample/es-plugin-reverse-ranking/compose.yaml`
- Create: `ir_sample/es-plugin-reverse-ranking/Makefile`
- Create: `ir_sample/es-plugin-reverse-ranking/scripts/verify.sh`

- [ ] **Step 1: Write the end-to-end verification script first**

The executable script must use `curl --fail-with-body` to recreate a one-shard `reverse-ranking-demo` index, bulk-index IDs and numeric ranks `0` through `4`, and query through this deterministic scorer:

```json
{
  "size": 3,
  "query": {
    "script_score": {
      "query": { "match_all": {} },
      "script": { "source": "doc['rank'].value" }
    }
  }
}
```

Add `"rescore":{"window_size":3,"reverse_rank":{}}` for the reranked request. Export the responses to Python 3 and assert:

```python
base = json.loads(os.environ["BASE_RESPONSE"])["hits"]["hits"]
reranked = json.loads(os.environ["RERANK_RESPONSE"])["hits"]["hits"]
zero_case = json.loads(os.environ["ZERO_RESPONSE"])["hits"]["hits"]
assert [hit["_id"] for hit in base] == ["4", "3", "2"]
assert [hit["_id"] for hit in reranked] == ["2", "3", "4"]
for hit, expected in zip(reranked, [0.5, 1.0 / 3.0, 0.25]):
    assert math.isclose(hit["_score"], expected, rel_tol=1e-6)
assert zero_case[-1]["_id"] == "0"
print("reverse ranking verification passed")
```

- [ ] **Step 2: Run the script to verify RED**

```bash
./scripts/verify.sh
```

Expected: curl fails because the plugin-enabled Elasticsearch service is not running.

- [ ] **Step 3: Add Docker and Make orchestration**

Use `gradle:8.10.2-jdk21` to build, then copy the ZIP into `docker.elastic.co/elasticsearch/elasticsearch:${ELASTICSEARCH_VERSION}` and install it:

```dockerfile
RUN bin/elasticsearch-plugin install --batch file:///tmp/reverse-ranking.zip \
    && rm /tmp/reverse-ranking.zip
```

`compose.yaml` must expose port 9200, disable security, use `-Xms512m -Xmx512m`, persist a named data volume, and wait on `/_cluster/health`. Add this Make interface:

```makefile
.PHONY: build up verify down logs

build:
	./gradlew clean test pluginZip

up:
	docker compose up --build --detach --wait

verify:
	./scripts/verify.sh

down:
	docker compose down --volumes --remove-orphans

logs:
	docker compose logs --follow elasticsearch
```

- [ ] **Step 4: Build and start Elasticsearch**

```bash
make build
make up
curl -fsS 'http://localhost:9200/_cat/plugins?h=component,version'
```

Expected: output contains `reverse-ranking 1.0.0`.

- [ ] **Step 5: Run verification to verify GREEN**

```bash
make verify
```

Expected: `reverse ranking verification passed`.

- [ ] **Step 6: Clean up and commit runtime support**

```bash
make down
git add ir_sample/es-plugin-reverse-ranking/.dockerignore
git add ir_sample/es-plugin-reverse-ranking/Dockerfile ir_sample/es-plugin-reverse-ranking/compose.yaml
git add ir_sample/es-plugin-reverse-ranking/Makefile ir_sample/es-plugin-reverse-ranking/scripts/verify.sh
git commit -m "test: verify reverse ranking in Elasticsearch"
```

### Task 5: Documentation And Final Verification

**Files:**
- Modify: `ir_sample/es-plugin-reverse-ranking/README.md`

- [ ] **Step 1: Replace the placeholder README**

Document Elasticsearch 8.19.21 and Java 21, all Make targets, a complete `reverse_rank` query, positive/zero score behavior, the generated ZIP, the plugin-list command, and the single-shard guarantee. State that normal Elasticsearch rescore behavior is per shard outside that guarantee.

- [ ] **Step 2: Run all automated verification**

```bash
make build
make up
make verify
make down
```

Expected: tests pass, Elasticsearch becomes healthy, verification passes, and cleanup succeeds.

- [ ] **Step 3: Check the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intended reverse-ranking files are listed.

- [ ] **Step 4: Commit documentation**

```bash
git add ir_sample/es-plugin-reverse-ranking/README.md
git commit -m "docs: explain reverse ranking plugin"
```