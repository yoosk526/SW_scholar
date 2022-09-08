const express = require("express");
const bodyParser = require("body-parser");

const server = express();
// CRUD ( Create, Read, Update, Delete )
server.use(bodyParser.json());

const users = [{
    "deviceName": "kk-A-Park\n",
    "deviceType": "IDLEPARK\n",
    "timestamp": 1662094009000,
    "payload": {
        "A1": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A2": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A3": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A4": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A5": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A6": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A7": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A8": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A9": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A10": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A11": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 },
        "A12": { "isfree": "False", "inTime": 1662094009000, "outTime": 0 }
    }
}
];

// server -> client
server.get("/api/user", (req, res) => {
    res.json(users);
});

// 특정 user id를 가져오는 경우
// 특정 주차 공간의 payload를 가져오는 경우
server.get("/api/user/:payload", (req, res) => {
    console.log(req.params.payload);
    const user = users.find((u) => {
        return u.payload === req.params.payload;
    });
    if (user) {    // 만약 불러올 payload가 user에 있다면 
        res.json(user);
    } else {
        res.status(404).json({ errorMessage: "Parking lot was not found" });
    }
});

// 새로운 user 추가
server.post("/api/user", (req, res) => {
    console.log(req.body);
    users.push(req.body);
    res.json(users);
});

// Update
server.put("/api/user/:payload", (req, res) => {
    let foundIndex = users.findIndex(u => u.payload === req.params.payload);
    if (foundIndex === -1) {    // Update할 주차 공간을 찾지 못한 경우
        res.status(404).json({ errorMessage: "Parking lot was not found" });
    } else {
        users[foundIndex] = { ...users[foundIndex], ...req.body };
        res.json(users[foundIndex]);
    }
});

// Delete
server.delete("/api/user/:payload", (req, res) => {
    let foundIndex = users.findIndex(u => u.payload === req.params.payload);
    if (foundIndex === -1) {    // Delete할 주차 공간를 찾지 못한 경우
        res.status(404).json({ errorMessage: "Parking lot was not found" });
    } else {
        let foundUser = users.splice(foundIndex, 1)    // 이 시작점부터 한칸을 지운다.
        res.json(foundUser[0]);
    }
});

server.listen(3000, () => {
    console.log("The server is running. local host = 3000");
});