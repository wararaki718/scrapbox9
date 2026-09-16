package com.example.elasticsearch.reverseranking;

import org.elasticsearch.plugins.SearchPlugin;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class ReverseRankingPluginTest {

    @Test
    void rescorer_spec_registered() {
        ReverseRankingPlugin plugin = new ReverseRankingPlugin();
        List<SearchPlugin.RescorerSpec<?>> specs = plugin.getRescorers();
        assertNotNull(specs);
        assertEquals(1, specs.size());

        SearchPlugin.RescorerSpec<?> spec = specs.get(0);
        assertEquals("reverse_rank", spec.getName().getPreferredName());
        assertNotNull(spec.getReader());
        assertNotNull(spec.getParser());
    }
}
