package utils

import (
	"container/list"
	"strings"
	"sync"

	"github.com/maximhq/bifrost/core/schemas"
)

const DefaultModelParamsCacheSize = 2048

// ModelParams holds cached parameters for a model.
// Add new fields here as more model-level parameters need caching.
type ModelParams struct {
	MaxOutputTokens         *int
	IsVertexMultiRegionOnly *bool // true when model is only available on Vertex multi-region pool endpoints (rep.googleapis.com)
}

type modelParamsCacheEntry struct {
	model  string
	params ModelParams
}

// inflightCall represents an in-progress cache miss handler invocation.
// Multiple goroutines waiting for the same model share one call.
type inflightCall struct {
	done   chan struct{}
	result *ModelParams
}

type modelParamsCache struct {
	mu               sync.RWMutex
	capacity         int
	items            map[string]*list.Element
	order            *list.List // front = most recently inserted/updated
	cacheMissHandler func(model string) *ModelParams

	inflightMu sync.Mutex
	inflight   map[string]*inflightCall
}

var (
	globalModelParamsCache *modelParamsCache
	cacheOnce              sync.Once
)

// knownAnthropicMaxOutputTokens provides static fallback defaults for Claude models
// when both cache and DB miss handler return nothing. Only Anthropic requires max_tokens.
var knownAnthropicMaxOutputTokens = map[string]int{
	"claude-fable-5":    128000,
	"claude-opus-4-6":   128000,
	"claude-sonnet-5":   128000,
	"claude-sonnet-4-6": 64000,
	"claude-haiku-4-5":  64000,
	"claude-sonnet-4-5": 64000,
	"claude-opus-4-5":   64000,
	"claude-opus-4-1":   32000,
	"claude-sonnet-4":   64000,
	"claude-opus-4":     32000,
	"claude-sonnet-4-0": 64000,
	"claude-opus-4-0":   32000,
	"claude-3-5-sonnet": 8192,
	"claude-3-5-haiku":  8192,
	"claude-3-7-sonnet": 8192,
	"claude-3-opus":     4096,
	"claude-3-sonnet":   4096,
	"claude-3-haiku":    4096,
}

func newModelParamsCache(capacity int) *modelParamsCache {
	return &modelParamsCache{
		capacity: capacity,
		items:    make(map[string]*list.Element, capacity),
		order:    list.New(),
		inflight: make(map[string]*inflightCall),
	}
}

func getModelParamsCache() *modelParamsCache {
	cacheOnce.Do(func() {
		globalModelParamsCache = newModelParamsCache(DefaultModelParamsCacheSize)
	})
	return globalModelParamsCache
}

func (c *modelParamsCache) Get(model string) (ModelParams, bool) {
	c.mu.Lock()
	elem, ok := c.items[model]
	if ok {
		c.order.MoveToFront(elem)
		params := elem.Value.(*modelParamsCacheEntry).params
		c.mu.Unlock()
		return params, true
	}
	handler := c.cacheMissHandler
	c.mu.Unlock()

	if handler == nil {
		return ModelParams{}, false
	}

	// Deduplicate concurrent miss handler calls for the same model.
	c.inflightMu.Lock()
	if call, ok := c.inflight[model]; ok {
		c.inflightMu.Unlock()
		<-call.done
		if call.result == nil {
			return ModelParams{}, false
		}
		return *call.result, true
	}
	call := &inflightCall{done: make(chan struct{})}
	c.inflight[model] = call
	c.inflightMu.Unlock()

	result := handler(model)
	call.result = result
	close(call.done)

	c.inflightMu.Lock()
	delete(c.inflight, model)
	c.inflightMu.Unlock()

	if result == nil {
		return ModelParams{}, false
	}
	c.Set(model, *result)
	return *result, true
}

func (c *modelParamsCache) Set(model string, params ModelParams) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[model]; ok {
		elem.Value.(*modelParamsCacheEntry).params = params
		c.order.MoveToFront(elem)
		return
	}

	if c.order.Len() >= c.capacity {
		c.evict()
	}

	entry := &modelParamsCacheEntry{model: model, params: params}
	elem := c.order.PushFront(entry)
	c.items[model] = elem
}

func (c *modelParamsCache) BulkSet(entries map[string]ModelParams) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for model, params := range entries {
		if elem, ok := c.items[model]; ok {
			elem.Value.(*modelParamsCacheEntry).params = params
			c.order.MoveToFront(elem)
			continue
		}

		if c.order.Len() >= c.capacity {
			c.evict()
		}

		entry := &modelParamsCacheEntry{model: model, params: params}
		elem := c.order.PushFront(entry)
		c.items[model] = elem
	}
}

func (c *modelParamsCache) evict() {
	tail := c.order.Back()
	if tail == nil {
		return
	}
	c.order.Remove(tail)
	delete(c.items, tail.Value.(*modelParamsCacheEntry).model)
}

func (c *modelParamsCache) Delete(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[model]; ok {
		c.order.Remove(elem)
		delete(c.items, model)
	}
}

// GetModelParams returns the cached parameters for a model.
// On cache miss, calls the registered miss handler (if any) to load from DB.
func GetModelParams(model string) (ModelParams, bool) {
	return getModelParamsCache().Get(model)
}

// SetModelParams sets the parameters for a model in the cache.
func SetModelParams(model string, params ModelParams) {
	getModelParamsCache().Set(model, params)
}

// BulkSetModelParams sets parameters for multiple models at once.
func BulkSetModelParams(entries map[string]ModelParams) {
	getModelParamsCache().BulkSet(entries)
}

// DeleteModelParams removes a model from the cache.
func DeleteModelParams(model string) {
	getModelParamsCache().Delete(model)
}

// SetCacheMissHandler registers a callback invoked on cache miss.
// The handler should query the DB for the model's parameters and return them,
// or nil if not found. The result is automatically cached.
func SetCacheMissHandler(fn func(model string) *ModelParams) {
	c := getModelParamsCache()
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cacheMissHandler = fn
}

// GetMaxOutputTokens returns the cached max_output_tokens for a model.
// Returns 0, false on cache miss or if max_output_tokens is not set.
func GetMaxOutputTokens(model string) (int, bool) {
	params, ok := GetModelParams(model)
	if !ok || params.MaxOutputTokens == nil {
		return 0, false
	}
	return *params.MaxOutputTokens, true
}

// GetMaxOutputTokensOrDefault returns the (provider, model)'s max_output_tokens
// from the resident capability table, or the provided default on miss. For Claude
// models it falls back to known static defaults before the caller's default. A
// legacy, non-provider-aware LRU lookup sits in between as a transitional fallback
// (still fed + lazily loaded) and is only reached when the capability table misses.
func GetMaxOutputTokensOrDefault(provider schemas.ModelProvider, model string, defaultValue int) int {
	if caps := CapabilitiesFor(provider, model); caps != nil && caps.MaxOutputTokens != nil {
		return *caps.MaxOutputTokens
	}
	if m, ok := GetMaxOutputTokens(model); ok {
		return m
	}
	if strings.Contains(model, "claude") {
		base := normalizeClaudeModelName(model)
		if base != model {
			if m, ok := GetMaxOutputTokens(base); ok {
				return m
			}
		}
		if m, ok := knownAnthropicMaxOutputTokens[base]; ok {
			return m
		}
	}
	return defaultValue
}

// IsVertexMultiRegionOnlyModel reports whether the given model is flagged in the
// datasheet as only available on Google Vertex multi-region pool endpoints
// (aiplatform.{region}.rep.googleapis.com). Returns false when the flag is not
// set. Reads the (model, Vertex) capability record, with a transitional fallback
// to the legacy LRU (keyed by the "vertex_ai/" provider-prefixed raw model).
func IsVertexMultiRegionOnlyModel(model string) bool {
	if caps := CapabilitiesFor(schemas.Vertex, model); caps != nil && caps.IsVertexMultiRegionOnly != nil {
		return *caps.IsVertexMultiRegionOnly
	}
	if params, ok := GetModelParams("vertex_ai/" + model); ok && params.IsVertexMultiRegionOnly != nil {
		return *params.IsVertexMultiRegionOnly
	}
	return false
}

// modelCapabilitiesTable holds per-(model, provider) bifrost overrides sourced
// from the datasheet model-parameters feed. Unlike the LRU model-params cache,
// this is a fully-resident map: the override set is tiny (a handful of curated
// models) yet read on the request hot path, so it must never be evicted. The
// datasheet sync swaps the whole map under the write lock via
// ReplaceModelCapabilities; reads take the shared lock. Keyed by CapabilityCacheKey
// ("<model>|<provider>").
var (
	modelCapabilitiesMu    sync.RWMutex
	modelCapabilitiesTable = map[string]*schemas.ModelCapabilities{}
)

// ReplaceModelCapabilities atomically swaps the entire overrides table. Called by
// the datasheet sync after parsing the model-parameters feed, which always
// carries the full set — so a wholesale replace also drops entries that no
// longer have overrides.
func ReplaceModelCapabilities(table map[string]*schemas.ModelCapabilities) {
	if table == nil {
		table = map[string]*schemas.ModelCapabilities{}
	}
	modelCapabilitiesMu.Lock()
	modelCapabilitiesTable = table
	modelCapabilitiesMu.Unlock()
}

// SetModelCapability additively inserts one override under the given cache key.
// The sync path uses ReplaceModelCapabilities; this is for single-entry callers
// and tests.
func SetModelCapability(cacheKey string, ov *schemas.ModelCapabilities) {
	modelCapabilitiesMu.Lock()
	modelCapabilitiesTable[cacheKey] = ov
	modelCapabilitiesMu.Unlock()
}

// DeleteModelCapability removes one override key (test cleanup).
func DeleteModelCapability(cacheKey string) {
	modelCapabilitiesMu.Lock()
	delete(modelCapabilitiesTable, cacheKey)
	modelCapabilitiesMu.Unlock()
}

// GetModelCapabilities returns the resident bifrost overrides stored under the
// exact cache key, or nil on miss. Overrides live under the CapabilityCacheKey
// composite ("<model>|<provider>"), so callers should almost always use
// CapabilitiesFor, which builds that key from the runtime
// (provider, model). This raw-key form is exported mainly for the sync path and
// tests.
func GetModelCapabilities(cacheKey string) *schemas.ModelCapabilities {
	modelCapabilitiesMu.RLock()
	ov := modelCapabilitiesTable[cacheKey]
	modelCapabilitiesMu.RUnlock()
	return ov
}

// CapabilityCacheKey builds the model-params cache key under which bifrost
// overrides are stored: "<model>|<provider>". This mirrors the pricing store's
// (model, provider) keying (datasheet makeKey), so a model's overrides stay
// distinct per provider — e.g. supports_speed=true on Anthropic vs absent on
// Vertex for the same claude-opus-4-8. The datasheet sync populates the cache
// with this exact key; CapabilitiesFor reconstructs it at request
// time from the runtime (provider, model).
func CapabilityCacheKey(model string, provider schemas.ModelProvider) string {
	return model + "|" + string(provider)
}

// CapabilitiesFor looks up bifrost overrides for a (provider,
// model) pair, mirroring how pricing resolves a (model, provider) row: overrides
// live under the CapabilityCacheKey composite, so provider is part of the key and
// the same model never collides across providers.
//
// Overrides are keyed by the datasheet's bare base_model (e.g. "claude-opus-4-7"),
// but the runtime model can be a Bedrock dotted id ("us.anthropic.claude-...-v1:0"),
// a Vertex "@version" id, or a dated variant. So after trying the exact model we
// collapse it to its base via normalizeClaudeModelName — the same normalization
// GetMaxOutputTokensOrDefault uses for Bedrock/Vertex — and try that. Plain
// schemas.BaseModelName is not enough: it strips date/version suffixes but not
// the provider/region prefix or ":"/"@" version markers.
//
// Returns nil on miss so callers can fall back to existing hardcoded helpers.
func CapabilitiesFor(provider schemas.ModelProvider, model string) *schemas.ModelCapabilities {
	if model == "" {
		return nil
	}
	if ov := GetModelCapabilities(CapabilityCacheKey(model, provider)); ov != nil {
		return ov
	}
	// normalizeClaudeModelName is Claude-specific — it strips everything before
	// the last ".", which mangles names like "gpt-4.1-2025-04-14" into "1".
	if strings.Contains(model, "claude") {
		if base := normalizeClaudeModelName(model); base != model {
			if ov := GetModelCapabilities(CapabilityCacheKey(base, provider)); ov != nil {
				return ov
			}
		}
	}
	return nil
}

// normalizeClaudeModelName extracts the base Claude model name from
// provider-specific model ID formats.
//
// Examples:
//
//	"claude-sonnet-4-20250514"                     → "claude-sonnet-4"
//	"anthropic.claude-sonnet-4-20250514-v1:0"      → "claude-sonnet-4"
//	"us.anthropic.claude-sonnet-4-20250514-v1:0"   → "claude-sonnet-4"
//	"claude-3-5-sonnet-20241022"                   → "claude-3-5-sonnet"
func normalizeClaudeModelName(model string) string {
	// Strip region + provider prefixes (us.anthropic., anthropic., etc.)
	if idx := strings.LastIndex(model, "."); idx >= 0 {
		model = model[idx+1:]
	}
	// Strip "@version" alias marker (Vertex/Bedrock, e.g. "...-4-5@20251001")
	if idx := strings.Index(model, "@"); idx >= 0 {
		model = model[:idx]
	}
	// Strip Bedrock version suffix (":0", ":1", etc.) and the preceding "-v1"/"-v2"
	if idx := strings.Index(model, ":"); idx >= 0 {
		model = model[:idx]
		if len(model) >= 3 {
			suffix := model[len(model)-3:]
			if suffix == "-v1" || suffix == "-v2" {
				model = model[:len(model)-3]
			}
		}
	}
	// Strip "-v1", "-v2" even without colon (e.g., "anthropic.claude-opus-4-6-v1")
	if strings.HasSuffix(model, "-v1") || strings.HasSuffix(model, "-v2") {
		model = model[:len(model)-3]
	}
	// Strip date version suffix using schemas.BaseModelName
	return schemas.BaseModelName(model)
}
