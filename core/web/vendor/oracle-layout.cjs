// Build a browser-safe API. The upstream worker is CommonJS, not a script tag.
module.exports = {
  layoutForceAtlas2: require('graphology-layout-forceatlas2'),
  FA2Layout: require('graphology-layout-forceatlas2/worker'),
};
