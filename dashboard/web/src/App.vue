<script setup lang="ts">
import { ref, onMounted } from "vue";
import { api, type Asset } from "./lib/api";
import AssetTab from "./components/AssetTab.vue";

const assets = ref<Asset[]>([]);
const activeId = ref<number | null>(null);
const error = ref<string | null>(null);

onMounted(async () => {
  try {
    const res = await api.getAssets();
    assets.value = res.assets;
    activeId.value = res.assets[0]?.id ?? null;
  } catch (e: any) {
    error.value = e.message;
  }
});
</script>

<template>
  <main>
    <h1>Elliott Wave Dashboard</h1>

    <p v-if="error" class="error">{{ error }}</p>

    <nav class="tabs">
      <button
        v-for="a in assets"
        :key="a.id"
        :class="{ active: a.id === activeId, disabled: a.status !== 'active' }"
        :disabled="a.status !== 'active'"
        @click="activeId = a.id"
      >
        {{ a.displayName }}
        <span v-if="a.status !== 'active'" class="soon">soon</span>
      </button>
    </nav>

    <AssetTab v-for="a in assets" v-show="a.id === activeId" :key="a.id" :asset="a" />
  </main>
</template>

<style scoped>
main {
  max-width: 960px;
  margin: 0 auto;
  padding: 1.5rem;
  font-family: system-ui, sans-serif;
}
.tabs {
  display: flex;
  gap: 0.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 0.5rem;
}
.tabs button {
  background: none;
  border: none;
  padding: 0.5rem 1rem;
  cursor: pointer;
  color: inherit;
  opacity: 0.6;
}
.tabs button.active {
  opacity: 1;
  border-bottom: 2px solid #58a6ff;
}
.tabs button.disabled {
  cursor: default;
}
.soon {
  font-size: 0.65rem;
  opacity: 0.5;
  margin-left: 4px;
}
.error {
  color: #f85149;
}
</style>
