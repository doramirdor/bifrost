package utils

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGetModelCapabilities_CacheHit verifies the lookup returns the stored
// override pointer for a known key.
func TestGetModelCapabilities_CacheHit(t *testing.T) {
	truePtr := true
	override := schemas.ModelCapabilities{
		SupportsCachePoint: &truePtr,
		ServerTools:        map[string]string{"web_search": "web_search_20260209"},
	}
	SetModelCapability("test/claude-opus-4-7", &override)
	t.Cleanup(func() { DeleteModelCapability("test/claude-opus-4-7") })

	got := GetModelCapabilities("test/claude-opus-4-7")
	require.NotNil(t, got)
	require.NotNil(t, got.SupportsCachePoint)
	assert.True(t, *got.SupportsCachePoint)
	assert.Equal(t, "web_search_20260209", got.ServerTools["web_search"])
}

// TestGetModelCapabilities_CacheMiss verifies missing keys return nil cleanly so
// callers can detect absence and fall back to hardcoded helpers.
func TestGetModelCapabilities_CacheMiss(t *testing.T) {
	got := GetModelCapabilities("test/nonexistent-model-12345")
	assert.Nil(t, got)
}

// TestGetModelCapabilities_IndependentOfModelParams verifies the override table
// is decoupled from the evictable LRU model-params cache: a model carrying only
// max_output_tokens has no override entry.
func TestGetModelCapabilities_IndependentOfModelParams(t *testing.T) {
	maxTokens := 4096
	SetModelParams("test/no-overrides-just-max-tokens", ModelParams{
		MaxOutputTokens: &maxTokens,
	})
	t.Cleanup(func() { DeleteModelParams("test/no-overrides-just-max-tokens") })

	got := GetModelCapabilities("test/no-overrides-just-max-tokens")
	assert.Nil(t, got)
}

// TestCapabilitiesFor_ProviderIsKey verifies the same model on
// different providers resolves to distinct overrides — provider is part of the
// key (mirrors pricing), so there is no cross-provider collision.
func TestCapabilitiesFor_ProviderIsKey(t *testing.T) {
	yes, no := true, false
	anthKey := CapabilityCacheKey("claude-opus-4-8", schemas.Anthropic)
	vertexKey := CapabilityCacheKey("claude-opus-4-8", schemas.Vertex)
	SetModelCapability(anthKey, &schemas.ModelCapabilities{SupportsFastMode: &yes})
	SetModelCapability(vertexKey, &schemas.ModelCapabilities{SupportsFastMode: &no})
	t.Cleanup(func() { DeleteModelCapability(anthKey); DeleteModelCapability(vertexKey) })

	anth := CapabilitiesFor(schemas.Anthropic, "claude-opus-4-8")
	require.NotNil(t, anth)
	assert.True(t, *anth.SupportsFastMode)

	vertex := CapabilitiesFor(schemas.Vertex, "claude-opus-4-8")
	require.NotNil(t, vertex)
	assert.False(t, *vertex.SupportsFastMode)
}

// TestCapabilitiesFor_NoCrossProviderCollision verifies a bare
// model on a non-Anthropic provider does NOT borrow the Anthropic-native entry —
// it misses cleanly so the caller falls back to substring detection.
func TestCapabilitiesFor_NoCrossProviderCollision(t *testing.T) {
	yes := true
	anthKey := CapabilityCacheKey("claude-opus-4-8", schemas.Anthropic)
	SetModelCapability(anthKey, &schemas.ModelCapabilities{SupportsFastMode: &yes})
	t.Cleanup(func() { DeleteModelCapability(anthKey) })

	assert.Nil(t, CapabilitiesFor(schemas.Vertex, "claude-opus-4-8"))
	assert.Nil(t, CapabilitiesFor(schemas.Bedrock, "claude-opus-4-8"))
	assert.Nil(t, CapabilitiesFor(schemas.Azure, "claude-opus-4-8"))
}

// TestCapabilitiesFor_BedrockDotted verifies a Bedrock dotted
// runtime id ("us.anthropic.claude-...-v1:0") collapses to the bare base_model
// the datasheet keys overrides under. This is exactly the case plain
// BaseModelName cannot handle (it leaves the provider/region prefix and
// ":version" intact), so it proves the stronger read-side normalization.
func TestCapabilitiesFor_BedrockDotted(t *testing.T) {
	yes := true
	// Stored under the bare base + bedrock provider, exactly as the sync keys it.
	key := CapabilityCacheKey("claude-opus-4-7", schemas.Bedrock)
	SetModelCapability(key, &schemas.ModelCapabilities{SupportsFastMode: &yes})
	t.Cleanup(func() { DeleteModelCapability(key) })

	got := CapabilitiesFor(schemas.Bedrock, "us.anthropic.claude-opus-4-7-20250101-v1:0")
	require.NotNil(t, got, "dotted Bedrock id should normalize to the base override entry")
	assert.True(t, *got.SupportsFastMode)
}

// TestCapabilitiesFor_VertexAtVersion verifies a Vertex
// "@version" runtime id collapses to the bare base_model.
func TestCapabilitiesFor_VertexAtVersion(t *testing.T) {
	yes := true
	key := CapabilityCacheKey("claude-haiku-4-5", schemas.Vertex)
	SetModelCapability(key, &schemas.ModelCapabilities{SupportsFastMode: &yes})
	t.Cleanup(func() { DeleteModelCapability(key) })

	got := CapabilitiesFor(schemas.Vertex, "claude-haiku-4-5@20251001")
	require.NotNil(t, got, "@version Vertex id should normalize to the base override entry")
	assert.True(t, *got.SupportsFastMode)
}

// TestCapabilitiesFor_BaseModelFallback verifies a dated model
// falls back to the base-model entry, matching the pricing capability lookup.
func TestCapabilitiesFor_BaseModelFallback(t *testing.T) {
	yes := true
	key := CapabilityCacheKey("claude-opus-4-7", schemas.Anthropic)
	SetModelCapability(key, &schemas.ModelCapabilities{SupportsFastMode: &yes})
	t.Cleanup(func() { DeleteModelCapability(key) })

	got := CapabilitiesFor(schemas.Anthropic, "claude-opus-4-7-20260401")
	require.NotNil(t, got, "dated model should fall back to base entry")
	assert.True(t, *got.SupportsFastMode)
}

// TestCapabilitiesFor_Miss verifies that an unknown
// (provider, model) pair returns nil cleanly.
func TestCapabilitiesFor_Miss(t *testing.T) {
	assert.Nil(t, CapabilitiesFor(schemas.Anthropic, "definitely-nonexistent-model-12345"))
}

// TestCapabilitiesFor_EmptyModel verifies short-circuit on empty
// model string.
func TestCapabilitiesFor_EmptyModel(t *testing.T) {
	assert.Nil(t, CapabilitiesFor(schemas.Anthropic, ""))
}

// TestReplaceModelCapabilities_DropsStale verifies the wholesale swap removes
// entries absent from the new table — the resync case where a model loses its
// overrides.
func TestReplaceModelCapabilities_DropsStale(t *testing.T) {
	yes := true
	key := CapabilityCacheKey("claude-opus-4-7", schemas.Anthropic)
	SetModelCapability(key, &schemas.ModelCapabilities{SupportsFastMode: &yes})
	require.NotNil(t, GetModelCapabilities(key))

	ReplaceModelCapabilities(map[string]*schemas.ModelCapabilities{})
	assert.Nil(t, GetModelCapabilities(key))
}

// TestOverrideAndMaxTokens_CoexistUnderSameKey verifies the override table and
// the max_output_tokens LRU are independent stores that can both answer for the
// same key.
func TestOverrideAndMaxTokens_CoexistUnderSameKey(t *testing.T) {
	maxTokens := 8192
	truePtr := true
	SetModelParams("test/combo-model", ModelParams{MaxOutputTokens: &maxTokens})
	SetModelCapability("test/combo-model", &schemas.ModelCapabilities{SupportsCachePoint: &truePtr})
	t.Cleanup(func() {
		DeleteModelParams("test/combo-model")
		DeleteModelCapability("test/combo-model")
	})

	gotMax, ok := GetMaxOutputTokens("test/combo-model")
	require.True(t, ok)
	assert.Equal(t, 8192, gotMax)

	gotOv := GetModelCapabilities("test/combo-model")
	require.NotNil(t, gotOv)
	require.NotNil(t, gotOv.SupportsCachePoint)
	assert.True(t, *gotOv.SupportsCachePoint)
}
