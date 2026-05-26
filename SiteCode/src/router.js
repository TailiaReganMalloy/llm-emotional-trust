import { createRouter, createWebHashHistory } from 'vue-router';

import BaseExplanationPage from '../baseExplain/baseExplanationPage.vue';
import StaticExplanationPage from '../staticExplain/staticExplanationPage.vue';
import StudyEntry from '../studyEntry.vue';
import HomePage from './pages/HomePage.vue';

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomePage,
  },
  {
    path: '/baseExplain',
    alias: ['/baseExplanation'],
    name: 'base-entry',
    component: StudyEntry,
    props: { studyType: 'base' },
  },
  {
    path: '/baseExplain/study',
    alias: ['/baseExplanation/study'],
    name: 'base-study',
    component: BaseExplanationPage,
  },
  {
    path: '/staticExplain',
    alias: ['/staticExplanation'],
    name: 'static-entry',
    component: StudyEntry,
    props: { studyType: 'static' },
  },
  {
    path: '/staticExplain/study',
    alias: ['/staticExplanation/study'],
    name: 'static-study',
    component: StaticExplanationPage,
  },
];

export default createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes,
});