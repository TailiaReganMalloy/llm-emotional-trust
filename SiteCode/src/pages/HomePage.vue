<template>
  <main class="home-page">
    <section class="hero-card">
      <p class="eyebrow">Local Study Preview</p>
      <h1>LLM Emotional Trust Experiment Sites</h1>
      <p class="intro">
        Run both study variants locally from one Vite dev server. Choose the experience you want to preview.
      </p>

      <nav class="link-grid">
        <RouterLink class="link-card" to="/staticExplain">
          <span class="link-label">Static Version</span>
          <strong>/staticExplain</strong>
          <p>Non-interactive explanation flow.</p>
        </RouterLink>

        <RouterLink class="link-card" to="/baseExplain">
          <span class="link-label">Interactive Version</span>
          <strong>/baseExplain</strong>
          <p>Interactive explanation flow with update logging.</p>
        </RouterLink>
      </nav>
    </section>
  </main>
</template>

<style scoped>
.home-page {
  min-height: 100vh;
  padding: 32px 20px;
  color: #1f2937;
  font-family: Georgia, 'Times New Roman', serif;
  background:
    radial-gradient(circle at top, rgb(250 233 201 / 0.8), transparent 40%),
    linear-gradient(180deg, #f8f4ea 0%, #ece6d8 100%);
}

.hero-card {
  max-width: 920px;
  margin: 0 auto;
  padding: 40px;
  background: rgb(255 252 245 / 0.88);
  border: 1px solid rgb(120 98 64 / 0.18);
  border-radius: 28px;
  box-shadow: 0 18px 60px rgb(77 54 23 / 0.12);
}

.eyebrow {
  margin: 0 0 12px;
  color: #8c5a1e;
  font-weight: 700;
  font-size: 0.8rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: clamp(2.2rem, 5vw, 4.4rem);
  line-height: 0.95;
}

.intro {
  max-width: 56ch;
  margin: 18px 0 0;
  color: #4b5563;
  font-size: 1.05rem;
  line-height: 1.6;
}

.link-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 18px;
  margin-top: 34px;
}

.link-card {
  display: block;
  padding: 24px;
  color: inherit;
  text-decoration: none;
  background: #fff;
  border: 1px solid rgb(120 98 64 / 0.16);
  border-radius: 20px;
  transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}

.link-card:hover {
  border-color: rgb(140 90 30 / 0.38);
  box-shadow: 0 14px 30px rgb(77 54 23 / 0.12);
  transform: translateY(-2px);
}

.link-label {
  display: block;
  margin-bottom: 10px;
  color: #8c5a1e;
  font-weight: 700;
  font-size: 0.8rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.link-card strong {
  display: block;
  font-size: 1.25rem;
}

.link-card p {
  margin: 12px 0 0;
  color: #4b5563;
  line-height: 1.5;
}

@media (width <= 640px) {
  .hero-card {
    padding: 28px;
    border-radius: 22px;
  }
}
</style>