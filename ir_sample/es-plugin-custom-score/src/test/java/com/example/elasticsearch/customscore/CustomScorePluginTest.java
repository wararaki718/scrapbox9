package com.example.elasticsearch.customscore;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.script.ScoreScript;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class CustomScorePluginTest {

    @Test
    void registersCustomScoreScriptEngine() {
        var plugin = new CustomScorePlugin();

        var engine = plugin.getScriptEngine(Settings.EMPTY, List.of(ScoreScript.CONTEXT));

        assertInstanceOf(CustomScoreScriptEngine.class, engine);
    }
}