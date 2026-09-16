package com.example.elasticsearch.reverseranking;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Explanation;
import org.elasticsearch.search.rescore.Rescorer;
import org.elasticsearch.search.rescore.RescoreContext;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

final class ReverseRankingRescorer implements Rescorer {

    static final ReverseRankingRescorer INSTANCE = new ReverseRankingRescorer();

    /* package-visible for tests */
    static float reciprocal(float score) {
        if (!Float.isFinite(score)) {
            throw new IllegalArgumentException("score must be finite");
        }
        if (score < 0f) throw new IllegalArgumentException("score must be non-negative");
        if (score == 0f) return 0f;
        float recip = 1f / score;
        if (!Float.isFinite(recip)) {
            throw new IllegalArgumentException("reciprocal score must be finite: " + score);
        }
        return recip;
    }

    /* package-visible for tests */
    static void rescore(ScoreDoc[] docs, int window) {
        if (docs == null || docs.length == 0) return;
        int n = Math.min(window, docs.length);
        for (int i = 0; i < n; i++) {
            docs[i].score = reciprocal(docs[i].score);
        }
        Arrays.sort(docs, (a, b) -> {
            int cmp = Float.compare(b.score, a.score);
            if (cmp != 0) return cmp;
            return Integer.compare(a.doc, b.doc);
        });
    }

    @Override
    public TopDocs rescore(TopDocs topDocs, IndexSearcher searcher, RescoreContext context) throws IOException {
        if (topDocs == null) return null;
        if (topDocs.scoreDocs == null || topDocs.scoreDocs.length == 0) return topDocs;

        ScoreDoc[] docs = topDocs.scoreDocs;
        int window = context.getWindowSize();

        // record original first-window doc IDs before transforming
        int n = Math.min(window, docs.length);
        Set<Integer> ids = new HashSet<>();
        for (int i = 0; i < n; i++) ids.add(docs[i].doc);
        context.setRescoredDocs(ids);

        // perform transformation and sort
        rescore(docs, window);

        return topDocs;
    }

    @Override
    public Explanation explain(int docId, IndexSearcher searcher, RescoreContext context, Explanation originalExplanation) throws IOException {
        if (originalExplanation == null) return null;
        if (!context.isRescored(docId)) return originalExplanation;
        Number originalValue = originalExplanation.getValue();
        if (originalValue == null) return originalExplanation;
        double val = originalValue.doubleValue();
        float recip = reciprocal((float) val);
        return Explanation.match(Double.valueOf(recip), "reciprocal of the original score", originalExplanation);
    }
}
