import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Image,
  StyleSheet,
  ScrollView,
  Button,
  TouchableOpacity,
} from "react-native";

const HomeScreen = () => {
  const [selectedAnimal, setSelectedAnimal] = useState("Bird");

  const animals = {
    Dog: {
      name: "Dog",
      image:
        "https://raw.githubusercontent.com/Jordanlouie1/RealityWarp/refs/heads/main/RealityInterpreter/assets/images/dog.png",
        wikiLink: "https://en.wikipedia.org/wiki/Dog",
      description:
        "Dogs are domesticated mammals, not natural wild animals. They were originally bred from wolves. They have been bred by humans for a long time, and were the first animals ever to be domesticated. Today, some dogs are used as pets, others are used to help humans do their work. They are a popular pet because they are usually playful, friendly, loyal and listen to humans. Thirty million dogs in the United States are registered as pets. Dogs eat both meat and vegetables, often mixed together and sold in stores as dog food. Dogs often have jobs, including as police dogs, army dogs, assistance dogs, fire dogs, messenger dogs, hunting dogs, herding dogs, or rescue dogs.",
    },
    Elephant: {
      name: "Elephant",
      image:
        "https://raw.githubusercontent.com/Jordanlouie1/RealityWarp/refs/heads/main/RealityInterpreter/assets/images/elephant.png",
        wikiLink: "https://en.wikipedia.org/wiki/Elephant",
      description:
        "Elephants are large mammals of the family Elephantidae and the order Proboscidea. Three species are currently recognised: the African bush elephant, the African forest elephant, and the Asian elephant. Elephants are scattered throughout sub-Saharan Africa, South Asia, and Southeast Asia. Elephantidae is the only surviving family of the order Proboscidea; other, now extinct, families of the order include mammoths and mastodons. Previously, the order was considered to contain, along with Elephantidae, the mammoths, but following a 2007 study, it is now generally accepted that they are a separate order. Elephants are the largest living land animals on Earth today.",
    },
    Bird: {
      name: "Bird",
      image:
        "https://raw.githubusercontent.com/Jordanlouie1/RealityWarp/refs/heads/main/RealityInterpreter/assets/images/parrot.png",
        wikiLink: "https://en.wikipedia.org/wiki/Bird",
      description: "Birds, also known as Aves, are a group of endothermic vertebrates, characterised by feathers, toothless beaked jaws, the laying of hard-shelled eggs, a high metabolic rate, a four-chambered heart, and a strong yet lightweight skeleton. Birds live worldwide and range in size from the 5 cm (2 in) bee hummingbird to the 2.75 m (9 ft) ostrich. They rank as the world's most numerically-successful class of tetrapods, with approximately ten thousand living species, more than half of these being passerines, sometimes known as perching birds. Birds have wings whose development varies according to species; the only known groups without wings are the extinct moa and elephant birds. Wings, which evolved from forelimbs, gave birds the ability to fly, although further evolution has led to the loss of flight in some birds, including ratites, penguins, and diverse endemic island species.",
    },
    Snake: {
      name: "Snake",
      image:
        "https://raw.githubusercontent.com/Jordanlouie1/RealityWarp/refs/heads/main/RealityInterpreter/assets/images/snake.png",
        wikiLink: "https://en.wikipedia.org/wiki/Snake",
      description:
        "Snakes are elongated, legless, carnivorous reptiles of the suborder Serpentes. Like all other squamates, snakes are ectothermic, amniote vertebrates covered in overlapping scales. Many species of snakes have skulls with many more joints than their lizard ancestors, enabling them to swallow prey much larger than their heads with their highly mobile jaws. To accommodate their narrow bodies, snakes' paired organs (such as kidneys) appear one in front of the other instead of side by side, and most have only one functional lung. Some species retain a pelvic girdle with a pair of vestigial claws on either side of the cloaca.",
    },
  };

  const fetchCurrentAnimal = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/current_animal");
      const data = await response.json();
      console.log("Current animal:", data);
      if (animals[data.animal]) {
        setSelectedAnimal(data.animal);
      }
    } catch (error) {
      console.error("Error fetching current animal:", error);
    }
  };

  useEffect(() => {
    const intervalId = setInterval(fetchCurrentAnimal, 2000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>
          be an animal to see like an animal!
        </Text>
        <View style={styles.animalContainer}>
          {Object.keys(animals).map((animalKey) => (
            <TouchableOpacity
              key={animalKey}
              onPress={() => setSelectedAnimal(animalKey)}
            >
              <Image
                source={{ uri: animals[animalKey].image }}
                style={[
                  styles.animalImage,
                  selectedAnimal !== animalKey && styles.greyedOutImage,
                ]}
              />
            </TouchableOpacity>
          ))}
        </View>
      </View>
      <View style={styles.content}>
        <View style={styles.videoContainer}></View>



        <Image
          source={{ uri: "http://127.0.0.1:5000/video_feed" }}
          style={styles.videoFeed}
        />
        <View style={styles.cardOverlay}>
          <View style={styles.card}>
            <Text style={styles.cardTitle}>{selectedAnimal}</Text>
            <Text style={styles.cardDescription}>
              {animals[selectedAnimal].description}
            </Text>
            <View style={styles.cardActions}>
              <Button
              title="Learn More"
              onPress={() => {
                window.open(animals[selectedAnimal].wikiLink, "_blank");
              }}
              />
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  header: {
    backgroundColor: "#949494",
    padding: 16,
  },
  headerText: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#fff",
    textAlign: "center",
  },
  animalContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginTop: 16,
  },
  animalImage: {
    width: 60,
    height: 60,
  },
  greyedOutImage: {
    opacity: 0.2,
  },
  content: {
    padding: 16,
  },
  videoFeed: {
    width: "100%",
    height: "80vh",
    marginBottom: 16,
  },
  card: {
    backgroundColor: "#fff",
    padding: 16,
    borderRadius: 8,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 2,
  },
  cardOverlay: {
    position: "absolute",
    right: 30,
    bottom: 70,
    width: "30%",
    justifyContent: "center",
    alignItems: "center",
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 14,
    color: "#666",
    marginBottom: 16,
  },
  cardActions: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
});

export default HomeScreen;
