package com.example.elasticsearch.reverseranking;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.search.rescore.RescorerBuilder;
import org.elasticsearch.search.rescore.RescoreContext;

import java.io.IOException;

public final class ReverseRankingRescorerBuilder extends RescorerBuilder<ReverseRankingRescorerBuilder> {

    public static final String NAME = "reverse_rank";
    private static final String WINDOW_SIZE_ERROR = "window_size must be greater than 0";

    public ReverseRankingRescorerBuilder() {
        super();
    }

    public ReverseRankingRescorerBuilder(StreamInput in) throws IOException {
        super(in);
    }

    public static ReverseRankingRescorerBuilder fromXContent(XContentParser parser) throws IOException {
        if (parser.currentToken() == null) {
            parser.nextToken();
        }
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(parser.getTokenLocation(), NAME + " expects an object");
        }

        XContentParser.Token token = parser.nextToken();
        if (token != XContentParser.Token.END_OBJECT) {
            String detail = token == XContentParser.Token.FIELD_NAME
                ? "field [" + parser.currentName() + "]"
                : "token [" + token + "]";
            throw new ParsingException(parser.getTokenLocation(), NAME + " does not support " + detail);
        }

        return new ReverseRankingRescorerBuilder();
    }

    @Override
    public ReverseRankingRescorerBuilder windowSize(int windowSize) {
        if (windowSize <= 0) {
            throw new IllegalArgumentException(WINDOW_SIZE_ERROR);
        }
        return super.windowSize(windowSize);
    }

    @Override
    protected boolean isWindowSizeRequired() {
        return true;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.endObject();
    }

    @Override
    protected RescoreContext innerBuildContext(int windowSize, org.elasticsearch.index.query.SearchExecutionContext context) {
        if (windowSize <= 0) {
            throw new IllegalArgumentException(WINDOW_SIZE_ERROR);
        }
        return new RescoreContext(windowSize, ReverseRankingRescorer.INSTANCE);
    }

    @Override
    public RescorerBuilder<ReverseRankingRescorerBuilder> rewrite(org.elasticsearch.index.query.QueryRewriteContext context) throws IOException {
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
