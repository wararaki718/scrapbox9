package com.example.elasticsearch.customscore;

import org.elasticsearch.script.ScoreScript;
import org.elasticsearch.script.UpdateScript;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CustomScoreScriptEngineTest {

    @Test
    void multipliesScoreBy100() {
        assertEquals(125.0, CustomScoreScriptEngine.multiplyBy100(1.25));
        assertEquals(0.0, CustomScoreScriptEngine.multiplyBy100(0.0));
    }

    @Test
    void exposesLanguageAndScoreContext() {
        var engine = new CustomScoreScriptEngine();

        assertEquals("custom_score", engine.getType());
        assertEquals(Set.of(ScoreScript.CONTEXT), engine.getSupportedContexts());
    }

    @Test
    void compilesSupportedScoreScript() {
        var engine = new CustomScoreScriptEngine();

        Object factory = engine.compile(null, "multiply_by_100", ScoreScript.CONTEXT, Map.of());

        assertInstanceOf(ScoreScript.Factory.class, factory);
    }

    @Test
    void rejectsUnknownSource() {
        var engine = new CustomScoreScriptEngine();

        var error = assertThrows(
            IllegalArgumentException.class,
            () -> engine.compile(null, "unknown", ScoreScript.CONTEXT, Map.of())
        );

        assertTrue(error.getMessage().contains("Unknown script source"));
    }

    @Test
    void rejectsUnsupportedContext() {
        var engine = new CustomScoreScriptEngine();

        var error = assertThrows(
            IllegalArgumentException.class,
            () -> engine.compile(null, "multiply_by_100", UpdateScript.CONTEXT, Map.of())
        );

        assertTrue(error.getMessage().contains("Unsupported script context"));
    }
}