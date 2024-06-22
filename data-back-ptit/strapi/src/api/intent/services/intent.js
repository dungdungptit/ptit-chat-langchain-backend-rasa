'use strict';

/**
 * intent service
 */

const { createCoreService } = require('@strapi/strapi').factories;

module.exports = createCoreService('api::intent.intent');
