package com.example.elasticsearch.reverseranking;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.io.stream.BytesStreamOutput;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.xcontent.DeprecationHandler;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.search.rescore.RescoreContext;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ReverseRankingRescorerBuilderTest {

    @Test
    void fromXContent_accepts_only_empty_object() throws Exception {
        try (XContentParser parser = createParser("{}")) {
            assertInstanceOf(ReverseRankingRescorerBuilder.class, ReverseRankingRescorerBuilder.fromXContent(parser));
        }

        try (XContentParser parser = createParser("{\"unknown\":true}")) {
            assertThrows(ParsingException.class, () -> ReverseRankingRescorerBuilder.fromXContent(parser));
        }

        try (XContentParser parser = createParser("[]")) {
            assertThrows(ParsingException.class, () -> ReverseRankingRescorerBuilder.fromXContent(parser));
        }

        try (XContentParser parser = createParser("\"value\"")) {
            assertThrows(ParsingException.class, () -> ReverseRankingRescorerBuilder.fromXContent(parser));
        }
    }

    @Test
    void windowSize_validation_and_stream_roundtrip() throws Exception {
        ReverseRankingRescorerBuilder builder = new ReverseRankingRescorerBuilder();

        IllegalArgumentException zero = assertThrows(IllegalArgumentException.class, () -> builder.windowSize(0));
        assertEquals("window_size must be greater than 0", zero.getMessage());
        IllegalArgumentException negative = assertThrows(IllegalArgumentException.class, () -> builder.windowSize(-1));
        assertEquals("window_size must be greater than 0", negative.getMessage());

        ReverseRankingRescorerBuilder original = new ReverseRankingRescorerBuilder().windowSize(3);

        BytesStreamOutput out = new BytesStreamOutput();
        out.setTransportVersion(TransportVersion.current());
        original.writeTo(out);
        StreamInput in = out.bytes().streamInput();
        in.setTransportVersion(TransportVersion.current());
        ReverseRankingRescorerBuilder copy = new ReverseRankingRescorerBuilder(in);
        assertEquals(3, copy.windowSize());
    }

    @Test
    void toXContent_writes_single_window_size_and_empty_reverse_rank() throws Exception {
        ReverseRankingRescorerBuilder builder = new ReverseRankingRescorerBuilder().windowSize(3);
        org.elasticsearch.xcontent.XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        ((ToXContent) builder).toXContent(xContentBuilder, ToXContent.EMPTY_PARAMS);
        BytesReference bytes = BytesReference.bytes(xContentBuilder);

        try (XContentParser parser = createParser(bytes.utf8ToString())) {
            assertEquals(XContentParser.Token.START_OBJECT, parser.nextToken());
            assertEquals(XContentParser.Token.FIELD_NAME, parser.nextToken());
            assertEquals("window_size", parser.currentName());
            assertEquals(XContentParser.Token.VALUE_NUMBER, parser.nextToken());
            assertEquals(3, parser.intValue());
            assertEquals(XContentParser.Token.FIELD_NAME, parser.nextToken());
            assertEquals("reverse_rank", parser.currentName());
            assertEquals(XContentParser.Token.START_OBJECT, parser.nextToken());
            assertEquals(XContentParser.Token.END_OBJECT, parser.nextToken());
            assertEquals(XContentParser.Token.END_OBJECT, parser.nextToken());
            assertEquals(null, parser.nextToken());
        }
    }

    @Test
    void build_context_creates_rescore_context() {
        ReverseRankingRescorerBuilder builder = new ReverseRankingRescorerBuilder().windowSize(2);
        RescoreContext context = builder.innerBuildContext(2, null);

        assertEquals(2, context.getWindowSize());
        assertSame(ReverseRankingRescorer.INSTANCE, context.rescorer());
    }

    private static XContentParser createParser(String json) throws Exception {
        return XContentType.JSON.xContent().createParser(
            NamedXContentRegistry.EMPTY,
            DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
            new ByteArrayInputStream(json.getBytes(StandardCharsets.UTF_8))
        );
    }
}
