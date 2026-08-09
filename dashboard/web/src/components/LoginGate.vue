<script setup lang="ts">
import { ref } from "vue";
import { api } from "../lib/api";

const emit = defineEmits<{ authenticated: [] }>();
const password = ref("");
const error = ref<string | null>(null);
const loading = ref(false);

async function login() {
  loading.value = true;
  error.value = null;
  try {
    await api.login(password.value);
    emit("authenticated");
  } catch (e: any) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <form class="login" @submit.prevent="login">
    <h1>Elliott Wave Dashboard</h1>
    <label>Password <input v-model="password" type="password" autocomplete="current-password" autofocus /></label>
    <button :disabled="loading">{{ loading ? "Signing in…" : "Sign in" }}</button>
    <p v-if="error" class="error">{{ error }}</p>
  </form>
</template>

<style scoped>
.login { max-width: 360px; margin: 20vh auto; display: grid; gap: 1rem; font-family: system-ui, sans-serif; }
label { display: grid; gap: .4rem; }
input { padding: .6rem; }
button { padding: .6rem; }
.error { color: #f85149; }
</style>
