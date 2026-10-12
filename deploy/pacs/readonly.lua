-- Med-Ops PACS: viewer accounts are read-only.
-- Only the job server's account may change the PACS (upload / delete / modify). Everyone
-- else (the judges' viewer account) can browse and view: GET requests plus the few POST
-- endpoints Orthanc Explorer 2 and the Stone Web Viewer use to search.

local WRITER = os.getenv('MEDOPS_PACS_WRITER') or 'medops-api'

local READ_ONLY_POST = {
  ['/tools/find'] = true,
  ['/tools/count-resources'] = true,
  ['/tools/lookup'] = true,
}

function IncomingHttpRequestFilter(method, uri, ip, username, httpHeaders)
  if username == WRITER then
    return true
  end
  if method == 'GET' then
    return true
  end
  if method == 'POST' and READ_ONLY_POST[uri] then
    return true
  end
  print('read-only account: denied ' .. method .. ' ' .. uri)
  return false
end
