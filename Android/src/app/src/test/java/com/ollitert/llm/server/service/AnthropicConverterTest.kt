/*
 * Copyright 2025-2026 @NightMean (https://github.com/NightMean)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ollitert.llm.server.service

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class AnthropicConverterTest {

  private val json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
  }

  private fun parse(body: String) = AnthropicConverter.parseRequest(json, body)

  @Test
  fun translatesPlainTextMessage() {
    val req = parse("""{"model":"x","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}""")
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertEquals("x", internal.model)
    assertEquals(10, internal.max_tokens)
    assertEquals(1, internal.messages.size)
    assertEquals("user", internal.messages[0].role)
    assertEquals("hi", internal.messages[0].content.text)
  }

  @Test
  fun systemStringPrependsAsSystemMessage() {
    val req = parse("""{"model":"x","max_tokens":1,"system":"You are helpful.","messages":[{"role":"user","content":"hi"}]}""")
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertEquals("system", internal.messages[0].role)
    assertEquals("You are helpful.", internal.messages[0].content.text)
    assertEquals("user", internal.messages[1].role)
  }

  @Test
  fun systemAsArrayOfTextBlocksIsConcatenated() {
    val req = parse(
      """{"model":"x","max_tokens":1,"system":[{"type":"text","text":"A"},{"type":"text","text":"B"}],
        "messages":[{"role":"user","content":"hi"}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertEquals("A\n\nB", internal.messages[0].content.text)
  }

  @Test
  fun systemAsArrayWithNonTextBlockThrows() {
    val req = parse(
      """{"model":"x","max_tokens":1,"system":[{"type":"image","text":"X"}],
        "messages":[{"role":"user","content":"hi"}]}"""
    )
    val ex = assertThrows(AnthropicConversionError::class.java) {
      AnthropicConverter.toInternalChatRequest(req)
    }
    assertEquals("invalid_request_error", ex.errorType)
  }

  @Test
  fun multiBlockContentBuildsParts() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":[
        {"type":"text","text":"caption"},
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
      ]}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val msg = internal.messages.single()
    assertEquals(2, msg.content.parts.size)
    assertEquals("text", msg.content.parts[0].type)
    assertEquals("image_url", msg.content.parts[1].type)
    assertEquals("data:image/png;base64,AAAA", msg.content.parts[1].image_url?.url)
  }

  @Test
  fun urlImageSourceIsRejected() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":[
        {"type":"image","source":{"type":"url","url":"http://example.com/x.png"}}]}]}"""
    )
    val ex = assertThrows(AnthropicConversionError::class.java) {
      AnthropicConverter.toInternalChatRequest(req)
    }
    assertEquals("invalid_request_error", ex.errorType)
    assertTrue(ex.message.contains("URL"))
  }

  @Test
  fun documentBlockRejected() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":[
        {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AAAA"}}]}]}"""
    )
    val ex = assertThrows(AnthropicConversionError::class.java) {
      AnthropicConverter.toInternalChatRequest(req)
    }
    assertEquals("invalid_request_error", ex.errorType)
  }

  @Test
  fun thinkingEchoBlocksAreDroppedSilently() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[
        {"role":"assistant","content":[
          {"type":"thinking","thinking":"old","signature":"sig"},
          {"type":"text","text":"answer"}
        ]},
        {"role":"user","content":"next"}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val assistant = internal.messages.first { it.role == "assistant" }
    assertEquals("answer", assistant.content.text)
    assertTrue(assistant.tool_calls.isNullOrEmpty())
  }

  @Test
  fun toolUseBlockProducesToolCalls() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"assistant","content":[
        {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}
      ]}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val msg = internal.messages.single { it.role == "assistant" }
    val call = msg.tool_calls!!.single()
    assertEquals("toolu_1", call.id)
    assertEquals("get_weather", call.function.name)
    assertTrue(call.function.arguments.contains("\"city\""))
  }

  @Test
  fun toolResultStringContentBecomesToolMessage() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":[
        {"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}
      ]}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val toolMsg = internal.messages.single { it.role == "tool" }
    assertEquals("toolu_1", toolMsg.tool_call_id)
    assertEquals("sunny", toolMsg.content.text)
  }

  @Test
  fun toolResultArrayContentJoinsTextBlocks() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":[
        {"type":"tool_result","tool_use_id":"toolu_1","content":[
          {"type":"text","text":"line1"},{"type":"text","text":"line2"}
        ]}
      ]}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val toolMsg = internal.messages.single { it.role == "tool" }
    assertEquals("line1\nline2", toolMsg.content.text)
  }

  @Test
  fun toolResultIsErrorPrefixesContent() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":[
        {"type":"tool_result","tool_use_id":"toolu_1","is_error":"true","content":"boom"}
      ]}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val toolMsg = internal.messages.single { it.role == "tool" }
    assertEquals("[error] boom", toolMsg.content.text)
  }

  @Test
  fun cacheControlSilentlyDropped() {
    val body = """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":[
      {"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}
    ]}],"tools":[{"name":"t","description":"d","input_schema":{},"cache_control":{"type":"ephemeral"}}]}"""
    val req = parse(body)
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertEquals("hi", internal.messages.single().content.parts.single().text)
    assertEquals("t", internal.tools!!.single().function.name)
  }

  @Test
  fun missingMaxTokensThrows() {
    val req = parse("""{"model":"x","messages":[{"role":"user","content":"hi"}]}""")
    val ex = assertThrows(AnthropicConversionError::class.java) {
      AnthropicConverter.toInternalChatRequest(req)
    }
    assertEquals("invalid_request_error", ex.errorType)
    assertTrue(ex.message.contains("max_tokens"))
  }

  @Test
  fun emptyMessagesThrows() {
    val req = parse("""{"model":"x","max_tokens":1,"messages":[]}""")
    val ex = assertThrows(AnthropicConversionError::class.java) {
      AnthropicConverter.toInternalChatRequest(req)
    }
    assertEquals("invalid_request_error", ex.errorType)
  }

  @Test
  fun toolDefinitionTranslatesToOpenAiShape() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":"hi"}],
        "tools":[{"name":"get_weather","description":"d","input_schema":{"type":"object"}}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val tool = internal.tools!!.single()
    assertEquals("function", tool.type)
    assertEquals("get_weather", tool.function.name)
    assertEquals("d", tool.function.description)
    assertNotNull(tool.function.parameters)
  }

  @Test
  fun toolChoiceAutoMapsToString() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":"hi"}],
        "tool_choice":{"type":"auto"}}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertEquals("auto", (internal.tool_choice as JsonPrimitive).content)
  }

  @Test
  fun toolChoiceAnyMapsToRequired() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":"hi"}],
        "tool_choice":{"type":"any"}}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertEquals("required", (internal.tool_choice as JsonPrimitive).content)
  }

  @Test
  fun toolChoiceSpecificToolMapsToFunctionObject() {
    val req = parse(
      """{"model":"x","max_tokens":1,"messages":[{"role":"user","content":"hi"}],
        "tool_choice":{"type":"tool","name":"get_weather"}}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    val choice = internal.tool_choice as JsonObject
    assertEquals("function", choice["type"]!!.jsonPrimitive.content)
    assertEquals("get_weather", choice["function"]!!.jsonObject["name"]!!.jsonPrimitive.content)
  }

  @Test
  fun toolChoiceMissingPassesThroughAsNull() {
    val req = parse("""{"model":"x","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}""")
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertNull(internal.tool_choice)
  }

  @Test
  fun stopSequencesPassThrough() {
    val req = parse(
      """{"model":"x","max_tokens":1,"stop_sequences":["END","\n\n"],
        "messages":[{"role":"user","content":"hi"}]}"""
    )
    val internal = AnthropicConverter.toInternalChatRequest(req)
    assertEquals(listOf("END", "\n\n"), internal.stop)
  }

  @Test
  fun metadataAndServiceTierIgnored() {
    val req = parse(
      """{"model":"x","max_tokens":1,"metadata":{"user_id":"u"},"service_tier":"flex",
        "messages":[{"role":"user","content":"hi"}]}"""
    )
    AnthropicConverter.toInternalChatRequest(req)  // Must not throw
  }
}
