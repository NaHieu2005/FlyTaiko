import {test} from 'node:test';import assert from 'node:assert/strict';
import {sign,valid,equal,sameOrigin} from '../lib/auth.js';
test('owner token rejects tampering, expired tokens and wrong audience',()=>{
 const key='test-secret-not-for-production';const token=sign('gpu-worker',60,key);
 assert.equal(valid(token,'gpu-worker',key),true);
 assert.equal(valid(token,'web-owner',key),false);
 assert.equal(valid(token+'x','gpu-worker',key),false);
 assert.equal(valid(sign('gpu-worker',-1,key),'gpu-worker',key),false);
 assert.equal(valid(token,'gpu-worker','wrong'),false);
 assert.equal(valid(token,'gpu-worker',''),false);
});
test('cross-origin login is rejected',()=>{assert.equal(sameOrigin({headers:{origin:'https://evil.test'}}),false);assert.equal(equal('a','b'),false);});
