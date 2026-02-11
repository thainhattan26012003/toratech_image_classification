/**
 * Multi-language: load from folder locale/*.json (en, ja, vi)
 */
const LOCALE_DIR = 'locale';
const SUPPORTED_LANGS = ['en', 'ja', 'vi'];

const I18N = {};

const STORAGE_KEY = 'image-classification-lang';
let currentLang = localStorage.getItem(STORAGE_KEY) || 'en';
if (!SUPPORTED_LANGS.includes(currentLang)) currentLang = 'en';

function t(key) {
  const locale = I18N[currentLang] || I18N.en || {};
  return locale[key] != null ? locale[key] : (I18N.en && I18N.en[key]) || key;
}

function setLanguage(lang) {
  if (!SUPPORTED_LANGS.includes(lang) || !I18N[lang]) return;
  currentLang = lang;
  localStorage.setItem(STORAGE_KEY, lang);
  document.documentElement.lang = lang === 'ja' ? 'ja' : lang === 'vi' ? 'vi' : 'en';
  applyTranslations();
  if (typeof onLanguageChange === 'function') onLanguageChange();
}

function getLanguage() {
  return currentLang;
}

function applyTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(function (el) {
    const key = el.getAttribute('data-i18n');
    const text = t(key);
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      if (el.type === 'file' && el.getAttribute('data-i18n-placeholder')) el.placeholder = text;
      else if (el.getAttribute('data-i18n-placeholder')) el.placeholder = text;
    } else {
      el.textContent = text;
    }
  });
  document.querySelectorAll('[data-i18n-title]').forEach(function (el) {
    el.title = t(el.getAttribute('data-i18n-title'));
  });
  if (document.title !== undefined) {
    document.title = t('title');
  }
}

function loadLocale(lang) {
  const url = LOCALE_DIR + '/' + lang + '.json';
  return fetch(url)
    .then(function (res) {
      if (!res.ok) throw new Error('Locale failed: ' + lang);
      return res.json();
    })
    .then(function (data) {
      I18N[lang] = data;
      return data;
    });
}

function loadAllLocales() {
  return Promise.all(SUPPORTED_LANGS.map(loadLocale));
}

document.addEventListener('DOMContentLoaded', function () {
  loadAllLocales()
    .then(function () {
      document.documentElement.lang = currentLang === 'ja' ? 'ja' : currentLang === 'vi' ? 'vi' : 'en';
      applyTranslations();
    })
    .catch(function (err) {
      console.error('i18n: failed to load locale files', err);
    });
});
