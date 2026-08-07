<script setup lang="ts">
/**
 * One threshold layer's live proximity-to-signal readout.
 *
 * Deliberately direction-agnostic: the same component renders a "short
 * building" gauge when the layer is SEEKING_HIGH and a "long building"
 * gauge when SEEKING_LOW, for either the macro or micro layer. Nothing
 * here assumes which one it's showing — that's what makes it reusable
 * instead of two near-duplicate components.
 */
import { computed } from "vue";
import type { SignalTensionLayer } from "../lib/api";

const props = defineProps<{
  label: string; // "Macro · wall street" / "Micro · behavioral"
  accent: string; // css color, caller picks per layer
  data: SignalTensionLayer;
}>();

const directionCopy = computed(() =>
  props.data.direction === "long"
    ? { verb: "bottom", noun: "long" }
    : { verb: "top", noun: "short" }
);

const barsProgressPct = computed(() =>
  Math.min(100, Math.round((props.data.barsElapsed / props.data.barsRequired) * 100))
);

function fmtPrice(v: number): string {
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
</script>

<template>
  <div class="layer" :style="{ '--accent': accent }">
    <div class="layer-top">
      <span class="layer-name">{{ label }}</span>
      <span class="direction-tag" :class="data.direction">{{ directionCopy.noun }}</span>
    </div>

    <div class="seeking">
      watching for a <b>{{ directionCopy.verb }}</b> to confirm
    </div>

    <div class="bar-row">
      <div class="bar-label">
        <span>price move toward threshold</span>
        <span>{{ Math.round(data.priceProgressPct) }}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill price" :style="{ width: data.priceProgressPct + '%' }" />
      </div>
    </div>

    <div class="bar-row">
      <div class="bar-label">
        <span>bars since last pivot</span>
        <span>{{ data.barsElapsed }} / {{ data.barsRequired }} min</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill bars" :class="{ ready: data.barsReady }" :style="{ width: barsProgressPct + '%' }" />
      </div>
    </div>

    <div class="trigger-line">
      <span>confirms below</span>
      <span class="val">{{ fmtPrice(data.triggerPrice) }}</span>
    </div>

    <div v-if="data.lastPivot" class="last-pivot">
      last confirmed: {{ data.lastPivot.structureLabel ?? "—" }}
      <template v-if="data.lastPivot.magnitudePct != null">
        · {{ data.lastPivot.magnitudePct > 0 ? "+" : "" }}{{ data.lastPivot.magnitudePct.toFixed(1) }}%
      </template>
    </div>
  </div>
</template>

<style scoped>
.layer {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-left: 2px solid var(--accent);
  padding: 14px;
  margin-bottom: 14px;
  font-size: 0.85rem;
}
.layer-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.layer-name {
  font-family: ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
}
.direction-tag {
  font-family: ui-monospace, monospace;
  font-size: 0.65rem;
  padding: 2px 7px;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.direction-tag.long {
  background: rgba(63, 185, 80, 0.15);
  color: #3fb950;
}
.direction-tag.short {
  background: rgba(248, 81, 73, 0.15);
  color: #f85149;
}
.seeking {
  font-size: 0.8rem;
  opacity: 0.75;
  margin-bottom: 10px;
}
.seeking b {
  opacity: 1;
}
.bar-row {
  margin-bottom: 9px;
}
.bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  opacity: 0.65;
  margin-bottom: 4px;
  font-family: ui-monospace, monospace;
}
.bar-track {
  height: 5px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 3px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 3px;
  background: var(--accent);
}
.bar-fill.bars {
  background: rgba(255, 255, 255, 0.25);
}
.bar-fill.bars.ready {
  background: #3fb950;
}
.trigger-line {
  display: flex;
  justify-content: space-between;
  font-family: ui-monospace, monospace;
  font-size: 0.75rem;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.last-pivot {
  font-size: 0.7rem;
  opacity: 0.5;
  margin-top: 6px;
}
</style>
