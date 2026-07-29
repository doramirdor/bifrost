package configstore

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testSemanticConfig() *ComplexitySemanticConfig {
	return &ComplexitySemanticConfig{
		Provider:       "openai",
		EmbeddingModel: "text-embedding-3-small",
		Dimension:      1536,
	}
}

func testSemanticAnalyzerConfig() *ComplexityAnalyzerConfig {
	cfg := testComplexityAnalyzerConfig()
	cfg.Semantic = testSemanticConfig()
	return cfg
}

func TestComplexitySemanticConfigTimeoutDecoding(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    time.Duration
		wantErr bool
	}{
		{name: "duration string", payload: `{"timeout":"250ms"}`, want: 250 * time.Millisecond},
		{name: "number is milliseconds", payload: `{"timeout":250}`, want: 250 * time.Millisecond},
		{name: "absent keeps zero", payload: `{}`, want: 0},
		{name: "null keeps zero", payload: `{"timeout":null}`, want: 0},
		{name: "negative number rejected", payload: `{"timeout":-5}`, wantErr: true},
		{name: "bad string rejected", payload: `{"timeout":"soon"}`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg ComplexitySemanticConfig
			err := json.Unmarshal([]byte(tt.payload), &cfg)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, cfg.Timeout)
		})
	}
}

func TestComplexitySemanticConfigTimeoutMarshalRoundTrip(t *testing.T) {
	cfg := testSemanticConfig()
	cfg.Timeout = 250 * time.Millisecond

	data, err := json.Marshal(cfg)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"timeout":"250ms"`)

	var decoded ComplexitySemanticConfig
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, cfg.Timeout, decoded.Timeout)
}

func TestComplexitySemanticConfigNormalizedDefaults(t *testing.T) {
	normalized := testSemanticConfig().normalized()

	assert.Equal(t, DefaultComplexitySemanticTimeout, normalized.Timeout)
	assert.Equal(t, ComplexitySemanticFallbackLexical, normalized.Fallback)
	assert.Equal(t, ComplexitySemanticVectorStoreEmbedded, normalized.VectorStore)
	require.NoError(t, normalized.Validate())
}

func TestComplexitySemanticConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*ComplexitySemanticConfig)
	}{
		{name: "missing provider", mutate: func(c *ComplexitySemanticConfig) { c.Provider = "" }},
		{name: "missing embedding model", mutate: func(c *ComplexitySemanticConfig) { c.EmbeddingModel = " " }},
		{name: "dimension too small", mutate: func(c *ComplexitySemanticConfig) { c.Dimension = 1 }},
		{name: "unknown fallback", mutate: func(c *ComplexitySemanticConfig) { c.Fallback = "llm" }},
		{name: "unknown vector store", mutate: func(c *ComplexitySemanticConfig) { c.VectorStore = "pgvector" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testSemanticConfig()
			tt.mutate(cfg)
			require.Error(t, cfg.normalized().Validate())
		})
	}
}

func TestDecodeComplexityAnalyzerConfigSemanticRoundTrip(t *testing.T) {
	cfg := testSemanticAnalyzerConfig()
	cfg.ConfigHashes = ComplexityAnalyzerConfigHashes{
		TierBoundaries:   "tier-hash",
		SimpleKeywords:   "simple-hash",
		MediumKeywords:   "medium-hash",
		ComplexKeywords:  "complex-hash",
		SemanticSettings: "settings-hash",
	}
	cfg.EmbeddingFingerprint = "fingerprint-1"

	raw, err := encodeComplexityAnalyzerConfig(cfg.Normalized())
	require.NoError(t, err)
	assert.Contains(t, string(raw), `"_embedding_fingerprint":"fingerprint-1"`)

	decoded, err := DecodeComplexityAnalyzerConfig(raw)
	require.NoError(t, err)
	require.NotNil(t, decoded.Semantic)
	assert.Equal(t, cfg.Normalized().Semantic, decoded.Semantic)
	assert.Equal(t, cfg.ConfigHashes, decoded.ConfigHashes)
	assert.Equal(t, "fingerprint-1", decoded.EmbeddingFingerprint)
}

func TestDecodeComplexityAnalyzerConfigWithoutSemantic(t *testing.T) {
	raw, err := encodeComplexityAnalyzerConfig(testComplexityAnalyzerConfig().Normalized())
	require.NoError(t, err)

	decoded, err := DecodeComplexityAnalyzerConfig(raw)
	require.NoError(t, err)
	assert.Nil(t, decoded.Semantic)
	assert.Empty(t, decoded.EmbeddingFingerprint)
}

func TestGenerateComplexityAnalyzerConfigHashesSemantic(t *testing.T) {
	base := testSemanticAnalyzerConfig()
	baseHashes, err := GenerateComplexityAnalyzerConfigHashes(base)
	require.NoError(t, err)
	require.NotEmpty(t, baseHashes.SemanticSettings)

	// Keyword edits must not move the semantic settings hash: the shared lists
	// are tracked by the keyword section hashes.
	keywordEdit := testSemanticAnalyzerConfig()
	keywordEdit.Keywords.SimpleKeywords = append(keywordEdit.Keywords.SimpleKeywords, "weather")
	keywordHashes, err := GenerateComplexityAnalyzerConfigHashes(keywordEdit)
	require.NoError(t, err)
	assert.Equal(t, baseHashes.SemanticSettings, keywordHashes.SemanticSettings)
	assert.NotEqual(t, baseHashes.SimpleKeywords, keywordHashes.SimpleKeywords)

	// Semantic scalar edits must not move the keyword hashes.
	scalarEdit := testSemanticAnalyzerConfig()
	scalarEdit.Semantic.EmbeddingModel = "text-embedding-3-large"
	scalarHashes, err := GenerateComplexityAnalyzerConfigHashes(scalarEdit)
	require.NoError(t, err)
	assert.NotEqual(t, baseHashes.SemanticSettings, scalarHashes.SemanticSettings)
	assert.Equal(t, baseHashes.SimpleKeywords, scalarHashes.SimpleKeywords)

	// No semantic section means no semantic hash.
	plainHashes, err := GenerateComplexityAnalyzerConfigHashes(testComplexityAnalyzerConfig())
	require.NoError(t, err)
	assert.Empty(t, plainHashes.SemanticSettings)
}

func TestMergeComplexityAnalyzerConfigByHashesSemantic(t *testing.T) {
	fileConfig := func() *ComplexityAnalyzerConfig {
		cfg := testSemanticAnalyzerConfig()
		hashes, err := GenerateComplexityAnalyzerConfigHashes(cfg)
		require.NoError(t, err)
		cfg.ConfigHashes = hashes
		return cfg
	}

	t.Run("file adds semantic to base without one", func(t *testing.T) {
		base := testComplexityAnalyzerConfig()
		file := fileConfig()

		merged, err := MergeComplexityAnalyzerConfigByHashes(base, file)
		require.NoError(t, err)
		require.NotNil(t, merged.Semantic)
		assert.Equal(t, file.Normalized().Semantic, merged.Semantic)
		assert.Equal(t, file.ConfigHashes.SemanticSettings, merged.ConfigHashes.SemanticSettings)
	})

	t.Run("unchanged hash preserves DB edits", func(t *testing.T) {
		file := fileConfig()
		base := fileConfig()
		// Simulate a UI edit persisted after the last file sync.
		base.Semantic.EmbeddingModel = "runtime-model"

		merged, err := MergeComplexityAnalyzerConfigByHashes(base, file)
		require.NoError(t, err)
		assert.Equal(t, "runtime-model", merged.Semantic.EmbeddingModel)
	})

	t.Run("settings change replaces the semantic block", func(t *testing.T) {
		base := fileConfig()
		base.Semantic.Fallback = ComplexitySemanticFallbackNone

		file := fileConfig()
		file.Semantic.EmbeddingModel = "text-embedding-3-large"
		fileHashes, err := GenerateComplexityAnalyzerConfigHashes(file)
		require.NoError(t, err)
		file.ConfigHashes = fileHashes

		merged, err := MergeComplexityAnalyzerConfigByHashes(base, file)
		require.NoError(t, err)
		assert.Equal(t, "text-embedding-3-large", merged.Semantic.EmbeddingModel)
		assert.Equal(t, ComplexitySemanticFallbackLexical, merged.Semantic.Fallback)
		assert.Equal(t, fileHashes.SemanticSettings, merged.ConfigHashes.SemanticSettings)
	})

	t.Run("file without semantic preserves DB semantic", func(t *testing.T) {
		base := fileConfig()
		base.EmbeddingFingerprint = "fingerprint-1"

		file := testComplexityAnalyzerConfig()
		fileHashes, err := GenerateComplexityAnalyzerConfigHashes(file)
		require.NoError(t, err)
		file.ConfigHashes = fileHashes

		merged, err := MergeComplexityAnalyzerConfigByHashes(base, file)
		require.NoError(t, err)
		require.NotNil(t, merged.Semantic)
		assert.Equal(t, base.Normalized().Semantic, merged.Semantic)
		assert.Equal(t, base.ConfigHashes.SemanticSettings, merged.ConfigHashes.SemanticSettings)
		assert.Equal(t, "fingerprint-1", merged.EmbeddingFingerprint)
	})
}

func TestRDBConfigStore_ComplexityAnalyzerConfigSemanticPersistence(t *testing.T) {
	store := setupRDBTestStore(t)
	ctx := context.Background()

	cfg := testSemanticAnalyzerConfig()
	cfg.EmbeddingFingerprint = "fingerprint-1"
	require.NoError(t, store.UpdateComplexityAnalyzerConfig(ctx, cfg))

	got, err := store.GetComplexityAnalyzerConfig(ctx)
	require.NoError(t, err)
	require.NotNil(t, got.Semantic)
	assert.Equal(t, cfg.Normalized().Semantic, got.Semantic)
	assert.Equal(t, "fingerprint-1", got.EmbeddingFingerprint)

	// A UI-style write without a fingerprint must not wipe the stored one.
	update := testSemanticAnalyzerConfig()
	update.Semantic.EmbeddingModel = "text-embedding-3-large"
	require.NoError(t, store.UpdateComplexityAnalyzerConfig(ctx, update))

	got, err = store.GetComplexityAnalyzerConfig(ctx)
	require.NoError(t, err)
	assert.Equal(t, "text-embedding-3-large", got.Semantic.EmbeddingModel)
	assert.Equal(t, "fingerprint-1", got.EmbeddingFingerprint)
}
