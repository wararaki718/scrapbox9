package com.example.elasticsearch.reverseranking;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Explanation;
import org.elasticsearch.search.rescore.RescoreContext;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

class ReverseRankingRescorerTest {

    @Test
    void reciprocal_basic() {
        assertEquals(0.25f, ReverseRankingRescorer.reciprocal(4.0f), 1e-6f);
        assertEquals(0.0f, ReverseRankingRescorer.reciprocal(0.0f), 0.0f);
    }

    @Test
    void reciprocal_invalid() {
        assertThrows(IllegalArgumentException.class, () -> ReverseRankingRescorer.reciprocal(-1f));
        assertThrows(IllegalArgumentException.class, () -> ReverseRankingRescorer.reciprocal(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> ReverseRankingRescorer.reciprocal(Float.POSITIVE_INFINITY));
        // reciprocal of Float.MIN_VALUE overflows to Infinity in float; should be rejected
        assertThrows(IllegalArgumentException.class, () -> ReverseRankingRescorer.reciprocal(Float.MIN_VALUE));
    }

    @Test
    void rescore_window_and_sort() {
        ScoreDoc d40 = new ScoreDoc(40, 4.0f);
        ScoreDoc d30 = new ScoreDoc(30, 3.0f);
        ScoreDoc d20 = new ScoreDoc(20, 2.0f);
        ScoreDoc d10 = new ScoreDoc(10, 1.0f);

        ScoreDoc[] docs = new ScoreDoc[]{d40, d30, d20, d10};

        ReverseRankingRescorer.rescore(docs, 3);

        int[] expectedDocs = new int[]{10,20,30,40};
        double[] expectedScores = new double[]{1.0, 0.5, 1.0/3.0, 0.25};

        for (int i = 0; i < docs.length; i++) {
            assertEquals(expectedDocs[i], docs[i].doc);
            assertEquals(expectedScores[i], docs[i].score, 1e-6);
        }
    }

    @Test
    void explain_rescored_and_unrescored() throws Exception {
        ScoreDoc d40 = new ScoreDoc(40, 4.0f);
        ScoreDoc d30 = new ScoreDoc(30, 3.0f);
        ScoreDoc d20 = new ScoreDoc(20, 2.0f);
        ScoreDoc d10 = new ScoreDoc(10, 1.0f);

        ScoreDoc[] docs = new ScoreDoc[]{d40, d30, d20, d10};
        TopDocs topDocs = new TopDocs(new TotalHits(docs.length, TotalHits.Relation.EQUAL_TO), docs);

        RescoreContext ctx = new RescoreContext(3, ReverseRankingRescorer.INSTANCE);
        // mark doc20 as rescored
        ctx.setRescoredDocs(Collections.singleton(20));

        Explanation source = Explanation.match(2.0f, "original");

        Explanation e20 = ReverseRankingRescorer.INSTANCE.explain(20, null, ctx, source);
        assertNotNull(e20);
        assertEquals(0.5, e20.getValue().doubleValue(), 1e-6);
        assertTrue(e20.toString().contains("reciprocal of the original score"));

        // unrescored doc returns original explanation
        Explanation e40 = ReverseRankingRescorer.INSTANCE.explain(40, null, ctx, Explanation.match(4.0f, "original"));
        assertEquals(4.0, e40.getValue().doubleValue(), 1e-6);
    }

    @Test
    void null_or_empty_topdocs_unchanged() throws Exception {
        // null TopDocs
        assertNull(ReverseRankingRescorer.INSTANCE.rescore(null, null, null));

        // empty scoreDocs
        TopDocs empty = new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        TopDocs out = ReverseRankingRescorer.INSTANCE.rescore(empty, null, new RescoreContext(3, ReverseRankingRescorer.INSTANCE));
        assertSame(empty, out);
    }
}
