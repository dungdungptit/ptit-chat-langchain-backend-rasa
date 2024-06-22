'use strict';

/**
 * intent router
 */

const { createCoreRouter } = require('@strapi/strapi').factories;

module.exports = createCoreRouter('api::intent.intent');
