# NeuroOracle layout bundle

`../static/vendor/oracle-layout.js` exposes the existing `graphologyLibrary`
interface (`layoutForceAtlas2` and `FA2Layout`) in a browser-safe IIFE. The old
`worker.min.js` script URL was CommonJS and threw `require is not defined`;
the expected namespace was never created. This bundle uses the same pinned
ForceAtlas2 0.10.1 release, not a new layout or a change to graph evidence.

Rebuild from this directory with Node.js 18 or later:

```console
npm ci --ignore-scripts
npm run build
```

Commit the generated bundle together with its license notice. Npm dependencies
are build-time only; Python and desktop users load the checked-in static file.
The lockfile records package versions and integrity. Do not commit node_modules.
Graphology and Sigma retain their pre-existing versioned CDN script URLs; this
change does not claim that the entire explorer works without a network.

Upstream API and license sources:

- https://graphology.github.io/standard-library/layout-forceatlas2.html
- `node_modules/graphology-layout-forceatlas2/LICENSE.txt`
- `node_modules/graphology-utils/LICENSE.txt`

The layout worker receives only the displayed node-position/edge arrays. It does
not retrieve sources, rewrite evidence, call a model, or mutate the server graph.
