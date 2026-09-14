package com.example.elasticsearch.customscore;

import org.elasticsearch.script.DocReader;
import org.elasticsearch.script.ScoreScript;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.script.ScriptEngine;

import java.util.Map;
import java.util.Set;

final class CustomScoreScriptEngine implements ScriptEngine {

    static final String LANGUAGE = "custom_score";
    static final String SOURCE = "multiply_by_100";

    static double multiplyBy100(double score) {
        return score * 100.0;
    }

    @Override
    public String getType() {
        return LANGUAGE;
    }

    @Override
    public <FactoryType> FactoryType compile(
        String name,
        String code,
        ScriptContext<FactoryType> context,
        Map<String, String> params
    ) {
        if (context != ScoreScript.CONTEXT) {
            throw new IllegalArgumentException("Unsupported script context: " + context.name);
        }
        if (!SOURCE.equals(code)) {
            throw new IllegalArgumentException("Unknown script source: " + code);
        }

        ScoreScript.Factory factory = (scriptParams, lookup) -> new ScoreScript.LeafFactory() {
            @Override
            public boolean needs_score() {
                return true;
            }

            @Override
            public boolean needs_termStats() {
                return false;
            }

            @Override
            public ScoreScript newInstance(DocReader reader) {
                return new ScoreScript(scriptParams, lookup, reader) {
                    @Override
                    public double execute(ExplanationHolder explanation) {
                        return multiplyBy100(get_score());
                    }
                };
            }
        };
        return context.factoryClazz.cast(factory);
    }

    @Override
    public Set<ScriptContext<?>> getSupportedContexts() {
        return Set.of(ScoreScript.CONTEXT);
    }
}