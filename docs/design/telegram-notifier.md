# 알림봇 (#7) 구현 가이드 — 텔레그램

> `docs/IMPLEMENTATION_PLAN.md` §7을 구현한 뒤 작성한 가이드 문서. `docs/design/recommendation-explainer.md`와 같은 위치·형식.

## 목적

추천 설명 생성기(#6)가 만든 `Explanation`을 텔레그램 메시지로 사용자에게 전송한다 (PRD §3, §4).

## MVP-0 범위 (알림 전용)

`IMPLEMENTATION_PLAN.md` §7에 따라 **1차 목표는 알림 전용**이다. 승인/거부 버튼이나 콜백 처리는 하지 않는다 — 승인해도 트리거되는 주문이 없기 때문이다(주문 실행 에이전트 #8이 아직 없음). 승인/거부 흐름은 #8과 함께 2차 목표(MVP-1)에서 추가한다.

## 모듈 구조

```
src/auto_stock/notifier/
├── __init__.py
├── models.py         # TelegramCredentials(bot_token, chat_id)
├── credentials.py     # load_telegram_credentials() -> TelegramCredentials
└── telegram_bot.py    # send_notification(explanation, credentials) -> None, TelegramNotificationError
```

## 왜 `python-telegram-bot`이 아니라 `requests` 직접 호출인가

`python-telegram-bot` v20+는 async 전용 API다. 이 프로젝트의 나머지 모듈(규칙엔진, 리스크사이징, 설명생성기)은 모두 동기 함수로 작성돼 있고, 지금 단계에서 asyncio 이벤트 루프를 도입할 이유가 없다(YAGNI). 텔레그램 Bot API는 단순 REST API이므로, 이미 다른 데이터 소스(FinanceDataReader 등)를 통해 프로젝트에 간접 의존성으로 들어와 있는 `requests`로 `sendMessage` 엔드포인트를 직접 호출하는 편이 의존성도 가볍고 동기 스타일과도 일관된다.

## 크리덴셜 관리

`CLAUDE.md`/PRD §7.4 원칙대로 **환경 변수로만 관리**한다.

- `credentials.py`의 `load_telegram_credentials()`가 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`를 `os.environ`에서 읽는다. 하나라도 없으면 `KeyError`를 던진다 — `python/security.md`가 권장하는 "필수 시크릿은 값이 없으면 즉시 실패" 패턴을 그대로 따른다.
- `python-dotenv`의 `load_dotenv()`를 호출해 로컬 개발 시 `.env` 파일에서도 값을 읽을 수 있게 했다. `.env`는 이미 `.gitignore`에 포함돼 있다.
- `.env.example`에 실제 값 없이 변수명만 커밋해뒀다 — `cp .env.example .env` 후 값을 채우면 된다.
- `send_notification()`은 크리덴셜을 인자로 받는다(내부에서 직접 `os.environ`을 읽지 않음) — 테스트 시 `requests.post`만 모킹하면 되고, 실제 환경변수나 `.env` 파일과 무관하게 순수 함수처럼 검증 가능하다.

### `TELEGRAM_CHAT_ID` 확인 방법

1. 텔레그램에서 생성한 봇을 검색해 대화를 시작하고 아무 메시지나 보낸다.
2. 브라우저 또는 `curl`로 `https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates`를 호출한다.
3. 응답 JSON에서 `result[0].message.chat.id` 값을 확인해 `TELEGRAM_CHAT_ID`로 사용한다.

## 메시지 포맷과 에러 처리

- 메시지: `[매수]`/`[매도]` 접두어 + `Explanation.summary` 그대로
- 텔레그램 API 응답이 HTTP 에러이거나 `{"ok": false}`이면 `TelegramNotificationError`를 던진다 — 알림 실패를 조용히 삼키지 않는다(silent failure 금지). 에러 메시지에는 상태 코드/텔레그램이 준 설명 문자열만 담고, 토큰이나 chat_id는 포함하지 않는다.

## 직접 테스트하기

실제 텔레그램 서버까지 메시지가 도달하는지 사용자가 직접 확인할 수 있도록 `scripts/verify_telegram.py`를 제공한다.

1. `.env.example`을 `.env`로 복사하고 `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` 값을 채운다.
2. **프로젝트 루트**에서 실행한다 (`load_dotenv()`가 현재 작업 디렉터리 기준으로 `.env`를 찾는다):
   ```
   .venv/Scripts/python scripts/verify_telegram.py   # Windows
   .venv/bin/python scripts/verify_telegram.py        # macOS/Linux
   ```
3. `SUCCESS: 텔레그램 메시지 전송 성공`이 출력되면 텔레그램 앱에서 실제로 메시지가 도착했는지 확인한다. 이 스크립트는 토큰/chat_id 값 자체를 절대 출력하지 않는다.

**`FAILED`가 나오면 흔한 원인**:

| 에러 설명 | 원인 | 대처 |
|---|---|---|
| `Bad Request: chat not found` | `TELEGRAM_CHAT_ID`가 틀렸거나, 봇과 아직 대화를 시작하지 않음 | 위 "TELEGRAM_CHAT_ID 확인 방법"대로 봇에게 먼저 메시지를 보낸 뒤 `getUpdates`로 chat_id 재확인 |
| `Unauthorized` | `TELEGRAM_BOT_TOKEN` 오타/무효 | 봇파더(@BotFather)에서 토큰 재확인 |
| `환경변수 누락 'TELEGRAM_BOT_TOKEN'` 등 | `.env`에 값이 비어 있거나 프로젝트 루트가 아닌 곳에서 실행함 | `.env` 파일 위치와 값 확인 |

## 테스트 전략

- `test_credentials.py`: `monkeypatch.setenv`/`delenv`로 환경변수 유무에 따른 정상 로드·`KeyError`를 검증 — 실제 `.env` 파일이나 진짜 토큰이 전혀 필요 없다.
- `test_telegram_bot.py`: `requests.post`를 모킹해 URL/payload, BUY/SELL 접두어, 실패 응답 시 예외 발생을 검증 — 네트워크 호출이 실제로 나가지 않는다.
- 실제 텔레그램으로의 전송(수동 확인)은 자동화 테스트 범위 밖이다. 사용자가 `.env`에 실제 토큰/chat_id를 채운 뒤 직접 확인한다.

## API 키

**텔레그램 봇 토큰(`TELEGRAM_BOT_TOKEN`)과 chat ID(`TELEGRAM_CHAT_ID`)가 필요하다.** 봇 토큰은 이미 발급받았다는 전제로 작성했다. 두 값 모두 코드에 하드코딩하지 않고 `.env`(로컬) 또는 배포 환경의 시크릿 매니저로 관리한다.

## security-reviewer 리뷰 결과 반영

구현 직후 `security-reviewer` 서브에이전트 리뷰에서 CRITICAL/HIGH는 없었고("승인"), LOW 1건이 지적되어 반영했다:

- **`requests.post()`에 타임아웃 없음** — `api.telegram.org`가 응답하지 않으면 알림 전송이 무한정 멈춰 있을 수 있어(자체 가용성 리스크), `timeout=10`을 추가했다. 동시에 `requests.RequestException`(타임아웃/연결 실패 등)도 `TelegramNotificationError`로 통일해서 던지도록 `try/except`를 추가했다 — 이전에는 네트워크 예외가 raw `requests` 예외로 그대로 새어나갔다.

MEDIUM(정보성)으로 지적된 "텔레그램 API가 돌려주는 `description` 문자열에 `chat_id`가 간접적으로 포함될 수 있어, 이후 로깅/알림 연동 시 노출 범위에 유의해야 한다"는 지금 당장 코드 변경이 필요한 사항은 아니라 이번 라운드에서는 반영하지 않았다 — 봇 토큰 자체는 어떤 예외 메시지에도 포함되지 않는다는 점은 리뷰에서 확인됐다.
