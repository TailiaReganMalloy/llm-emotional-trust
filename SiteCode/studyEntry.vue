<template>
  <div class="study-entry-page">
    <div class="entry-wrapper">
      <h1 class="headline">Welcome to the Study</h1>
      
      <div class="entry-content">
        <p class="entry-description">
          Thank you for participating in this study. You can start immediately without
          entering an email address, username, or Prolific ID.
        </p>

        <div class="continue-section">
          <button 
            @click="continueToStudy"
            class="continue-button"
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router';

const props = defineProps({
  studyType: {
    type: String,
    default: null
  }
});

const router = useRouter();

function continueToStudy() {
  const userInfo = {
    email: null,
    prolificId: null,
  };
  
  sessionStorage.setItem('studyUserInfo', JSON.stringify(userInfo));
  
  // Navigate to the appropriate study page based on the prop
  if (props.studyType === 'static') {
    router.push('/staticExplain/study');
  } else if (props.studyType === 'base') {
    router.push('/baseExplain/study');
  } else if (props.studyType === 'text') {
    router.push('/textExplanation/study');
  } else {
    // Default fallback - shouldn't normally happen
    router.push('/');
  }
}
</script>

<style scoped>
.study-entry-page {
  min-height: 100vh;
  padding: 40px 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  background-color: #f5f5f5;
}

.entry-wrapper {
  max-width: 600px;
  margin: 0 auto;
  padding: 40px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgb(0 0 0 / 10%);
}

.headline {
  margin-bottom: 30px;
  color: #333;
  font-weight: 700;
  font-size: 28px;
  text-align: center;
}

.entry-description {
  margin-bottom: 30px;
  color: #555;
  font-size: 16px;
  line-height: 1.6;
  text-align: center;
}

.continue-section {
  padding-top: 20px;
  text-align: center;
}

.continue-button {
  width: 100%;
  padding: 16px 24px;
  color: white;
  font-weight: 600;
  font-size: 16px;
  background-color: #4a90d9;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.continue-button:hover {
  background-color: #3a7bc8;
  box-shadow: 0 4px 12px rgb(74 144 217 / 30%);
  transform: translateY(-2px);
}

@media (width <= 600px) {
  .entry-wrapper {
    padding: 24px;
  }

  .headline {
    font-size: 24px;
  }
}
</style>
