package com.example.elasticsearch.reverseranking;

import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;

import java.util.List;

public final class ReverseRankingPlugin extends Plugin implements SearchPlugin {

    @Override
    public List<SearchPlugin.RescorerSpec<?>> getRescorers() {
        return List.of(
            new SearchPlugin.RescorerSpec<>(
                ReverseRankingRescorerBuilder.NAME,
                ReverseRankingRescorerBuilder::new,
                ReverseRankingRescorerBuilder::fromXContent
            )
        );
    }
}
